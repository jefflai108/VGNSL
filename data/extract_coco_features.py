# Email  : maojiayuan@gmail.com
# Date   : 11/27/2018
#
# Distributed under terms of the MIT license.

import os.path as osp
import queue
import threading
import numpy as np

from PIL import Image

import torch
import torch.nn as nn
import torch.cuda as cuda
import torch.backends.cudnn as cudnn

from torch.utils.data.dataset import Dataset

import jacinle.io as io

from jacinle.cli.argument import JacArgumentParser
from jacinle.logging import get_logger
from jacinle.utils.container import GView
from jacinle.utils.tqdm import tqdm
from jactorch.cuda.copy import async_copy_to

logger = get_logger(__file__)
io.set_fs_verbose(True)

parser = JacArgumentParser()
parser.add_argument('--train-caption', required=True, type='checked_file', help='caption annotations (*.json)')
parser.add_argument('--val-caption', required=True, type='checked_file', help='caption annotations (*.json)')
parser.add_argument('--image-root', required=True, type='checked_dir', help='image directory')
parser.add_argument('--output', required=True, help='output .h5 file')

parser.add_argument('--image-size', default=224, type=int, metavar='N', help='input image size')
parser.add_argument('--batch-size', default=64, type=int, metavar='N', help='batch size')
parser.add_argument('--data-workers', type=int, default=4, metavar='N', help='the num of workers that input training data')

parser.add_argument('--use-gpu', type='bool', default=True, metavar='B', help='use GPU or not')
parser.add_argument('--force-gpu', action='store_true', help='force the script to use GPUs, useful when there exists on-the-ground devices')

parser.add_argument('--random-embed', action='store_true', help='random embeddings instead of passing through visual models')

args = parser.parse_args()
args.output_images_json = osp.splitext(args.output)[0] + '.images.json'

if args.use_gpu:
    nr_devs = cuda.device_count()
    if args.force_gpu and nr_devs == 0:
        nr_devs = 1
    assert nr_devs > 0, 'No GPU device available'
    args.gpus = [i for i in range(nr_devs)]
    args.gpu_parallel = (nr_devs > 1)


class COCOImageDataset(Dataset):
    def __init__(self, images, image_root, image_transform):
        self.images = images
        self.image_root = image_root
        self.image_transform = image_transform

    def __getitem__(self, index):
        feed_dict = GView()
        feed_dict.image_filename = self.images[index]
        if self.image_root is not None:
            feed_dict.image = Image.open(osp.join(self.image_root, feed_dict.image_filename)).convert('RGB')
            feed_dict.image = self.image_transform(feed_dict.image)

        return (feed_dict.raw(), feed_dict.image_filename)

    def __len__(self):
        return len(self.images)

    def make_dataloader(self, batch_size, shuffle, drop_last, nr_workers):
        from jactorch.data.dataloader import JacDataLoader
        from jactorch.data.collate import VarLengthCollateV2

        collate_guide = {
            'image_filename': 'skip',
        }

        return JacDataLoader(
            self, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last,
            num_workers=nr_workers, pin_memory=True,
            collate_fn=VarLengthCollateV2(collate_guide)
        )


class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        # standard pytorch resnet feature extraction code 
        model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet152', pretrained=True)
        self.resnet = torch.nn.Sequential(*(list(model.children())[:-1])) # take representation before last classification layer 
        self.resnet.eval()

    def forward(self, feed_dict, image_filename, random_embed=False, batch_size=64):
        feed_dict = GView(feed_dict)
        with torch.no_grad(): 
            if random_embed: # random embed
                print('random embed')
                f = torch.randn(batch_size, 2048)
            else: # default
                f = self.resnet(feed_dict.image).squeeze(-1).squeeze(-1)

            print(f.shape) # N, 2048

        f = f.cpu().detach().numpy() # for storing purpose 
        output_dict = {image_filename[i]: f[i] for i in range(len(image_filename))}
        
        return output_dict

def main(model, caption_file, output_dict = {}, random_embed=False, batch_size=64):
    logger.critical('Loading the dataset.')
    data = io.load(caption_file)
    # Step 1: filter out images.
    images = [pair['image'] for pair in data['data']]

    import torchvision.transforms as T
    image_transform = T.Compose([
        T.Resize((args.image_size, args.image_size)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    dataset = COCOImageDataset(images, args.image_root, image_transform)
    dataloader = dataset.make_dataloader(args.batch_size, shuffle=False, drop_last=False, nr_workers=args.data_workers)

    for feed_dict in tqdm(dataloader, total=len(dataloader), desc='Extracting features'):
        if args.use_gpu:
            feed_dict, image_filename = async_copy_to(feed_dict, 0)
            image_filename = [x.split('/')[1] for x in image_filename] 
            #image_filename = ['/'.join([x.split('/')[0].replace('2014', '2017'), x.split('/')[1].split('_')[2]]) for x in image_filename] # use the 2017 mscoco instead of 2014
        
        with torch.no_grad():
            output_dict.update(model(feed_dict, image_filename, random_embed, batch_size))
   
    return output_dict, len(images)
    
if __name__ == '__main__':
    output_file = io.open_h5(args.output, 'w')

    logger.critical('Building the model.')
    model = FeatureExtractor()
    if args.use_gpu:
        model.cuda()
        if args.gpu_parallel:
            from jactorch.parallel import JacDataParallel
            model = JacDataParallel(model, device_ids=args.gpus).cuda()
        cudnn.benchmark = True
    model.eval()

    output_dict = {}
    output_dict, val_image_len = main(model, args.val_caption, output_dict, args.random_embed, args.batch_size)
    output_dict, train_image_len = main(model, args.train_caption, output_dict, args.random_embed, args.batch_size)

    assert len(output_dict) == val_image_len + train_image_len
    for k, v in output_dict.items():
        output_file.create_dataset(k, data=np.array(v))
    
    output_file.close()

