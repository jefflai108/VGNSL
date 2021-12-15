import numpy as np 

import benepar
benepar.download('benepar_en3')

# load text captions 
captions = list()
with open('data/mscoco/dev_caps.txt', 'r') as f: 
    for line in f: 
        print(line)
        captions.append(line.strip('\n').lower().split())
length = len(captions) 

# load image features 
images = np.load('data/mscoco/dev_ims.npy')
print(images.shape) # (1000, 2048)

# create golden parse tree

class PrecompDataset(data.Dataset):
    """ load precomputed captions and image features """

    def __init__(self, data_path, data_split, vocab, 
                 load_img=True, img_dim=2048):
        self.vocab = vocab

        # captions
        self.captions = list()
        with open(os.path.join(data_path, f'{data_split}_caps.txt'), 'r') as f:
            for line in f:
                self.captions.append(line.strip().lower().split())
            f.close()
        self.length = len(self.captions)

        # image features
        if load_img:
            self.images = np.load(os.path.join(data_path, f'{data_split}_ims.npy'))
        else:
            self.images = np.zeros((self.length // 5, img_dim))
        
        # each image can have 1 caption or 5 captions 
        if self.images.shape[0] != self.length:
            self.im_div = 5
            assert self.images.shape[0] * 5 == self.length
        else:
            self.im_div = 1

    def __getitem__(self, index):
        # image
        img_id = index  // self.im_div
        image = torch.tensor(self.images[img_id])
        # caption
        caption = [self.vocab(token) 
                   for token in ['<start>'] + self.captions[index] + ['<end>']]
        caption = torch.tensor(caption)
        return image, caption, index, img_id

    def __len__(self):
        return self.length


