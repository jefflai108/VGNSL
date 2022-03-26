from PIL import Image
import requests

from torchvision import transforms as pth_transforms
import torch

def get_image(url):
  return Image.open(requests.get(url, stream=True).raw)

preprocess = pth_transforms.Compose([
        pth_transforms.Resize(256, interpolation=3),
        pth_transforms.CenterCrop(224),
        pth_transforms.ToTensor(),
        pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

def get_features(model, image):
  return model(preprocess(image).unsqueeze(0))

resnet50 = torch.hub.load('facebookresearch/dino:main', 'dino_resnet50')
vits8 = torch.hub.load('facebookresearch/dino:main', 'dino_vits8')

url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
x = get_image(url)
print(preprocess(x).shape) # torch.Size([3, 224, 224])
features = get_features(resnet50, x)
features = get_features(vits8, x)
print(features.shape) # torch.Size([1, 2048]) or torch.Size([1, 384])
