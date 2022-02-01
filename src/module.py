import torch 
import torch.nn as nn 
import math
import numpy as np

class AttentivePooling(nn.Module):
    """
    Attentive Pooling module incoporate attention mask 
    """
    def __init__(self, feature_dim, output_dim, **kwargs):
        super(AttentivePooling, self).__init__()
        
        self.feature_transform = nn.Linear(feature_dim, output_dim)
        self.W_a = nn.Linear(output_dim, output_dim)
        self.W = nn.Linear(output_dim, 1)
        self.act_fn = nn.ReLU()
        self.softmax = nn.functional.softmax

    def forward(self, batch_rep, att_mask):
        """
        input:
        batch_rep : size (B, T, H), B: batch size, T: sequence length, H: Hidden dimension
        att_mask:  size (B, T),     Attention Mask logits
        
        attention_weight:
        att_w : size (B, T, 1)
        
        return:
        utter_rep: size (B, H)
        """
        batch_rep  = self.feature_transform(batch_rep)
        att_logits = self.W(self.act_fn(self.W_a(batch_rep))).squeeze(-1)
        att_logits = att_mask + att_logits # masked out frames recieves ~0% prob. 
        # compute attention att_w
        # softmax over segment dimension i.e. take the most representation frame to represent word 
        att_w = self.softmax(att_logits, dim=-1).unsqueeze(-1)
        # apply att_w to input
        segment_rep = torch.sum(batch_rep * att_w, dim=1) 

        return segment_rep

class AttentivePoolingInputNorm(nn.Module):
    """
    Attentive Pooling module incoporate attention mask 
    """
    def __init__(self, feature_dim, output_dim, **kwargs):
        super(AttentivePoolingInputNorm, self).__init__()
       
        self.input_feature_transform = nn.Sequential(
            nn.Linear(feature_dim, feature_dim), 
            nn.Dropout(p=0.2),
            nn.ReLU(), 
            nn.Linear(feature_dim, output_dim), 
            nn.Dropout(p=0.2),
            nn.ReLU(), 
        )

        self.W_a = nn.Linear(output_dim, output_dim)
        self.W = nn.Linear(output_dim, 1)
        self.act_fn = nn.ReLU()
        self.softmax = nn.functional.softmax

    def forward(self, batch_rep, att_mask):
        """
        input:
        batch_rep : size (B, T, H), B: batch size, T: sequence length, H: Hidden dimension
        att_mask:  size (B, T),     Attention Mask logits
        
        attention_weight:
        att_w : size (B, T, 1)
        
        return:
        utter_rep: size (B, H)
        """
        batch_rep  = self.input_feature_transform(batch_rep)
        att_logits = self.W(self.act_fn(self.W_a(batch_rep))).squeeze(-1)
        att_logits = att_mask + att_logits # masked out frames recieves ~0% prob. 
        # compute attention att_w
        # softmax over segment dimension i.e. take the most representation frame to represent word 
        att_w = self.softmax(att_logits, dim=-1).unsqueeze(-1)
        # apply att_w to input
        segment_rep = torch.sum(batch_rep * att_w, dim=1) 

        return segment_rep

    def _norm(self, x): 
        x_mean = torch.mean(x, dim=0)
        x_std = torch.std(x, dim=0)
        return (x - x_mean) / x_std

################################
def conv1d(in_planes, out_planes, width=9, stride=1, bias=False):
    """1xd convolution with padding"""
    if width % 2 == 0:
        pad_amt = int(width / 2)
    else:
        pad_amt = int((width - 1) / 2)
    return nn.Conv2d(in_planes, out_planes, kernel_size=(1,width), 
                     stride=stride, padding=(0,pad_amt), bias=bias)

class SpeechBasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, width=9, stride=1, downsample=None):
        super(SpeechBasicBlock, self).__init__()
        self.conv1 = conv1d(inplanes, planes, width=width, stride=stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv1d(planes, planes, width=width)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        print('entering SpeechBasicBlock')
        print(x.shape)
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        print(out.shape)
        out += residual
        print(out.shape)
        out = self.relu(out)
        return out

class ResDavenet(nn.Module):
    def __init__(self, feat_dim=40, block=SpeechBasicBlock, layers=[2, 2, 2, 2],
                 layer_widths=[128, 128, 256, 512, 1024], convsize=9):
        assert(len(layers) == 4)
        assert(len(layer_widths) == 5)
        super(ResDavenet, self).__init__()
        self.feat_dim = feat_dim
        self.inplanes = layer_widths[0]
        self.batchnorm1 = nn.BatchNorm2d(1)
        self.conv1 = nn.Conv2d(1, self.inplanes, kernel_size=(self.feat_dim,1), 
                               stride=1, padding=(0,0), bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, layer_widths[1], layers[0], 
                                       width=convsize, stride=2)
        self.layer2 = self._make_layer(block, layer_widths[2], layers[1], 
                                       width=convsize, stride=2)
        self.layer3 = self._make_layer(block, layer_widths[3], layers[2], 
                                       width=convsize, stride=2)
        self.layer4 = self._make_layer(block, layer_widths[4], layers[3], 
                                       width=convsize, stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, width=9, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                        kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )       
        layers = []
        layers.append(block(self.inplanes, planes, width=width, stride=stride, 
                            downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, width=width, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        if x.dim() == 3: # create channel dimension
            x = x.unsqueeze(1)
        print(x.shape) # torch.Size([270, 1, 60, 40])
        x = self.conv1(x)
        print(x.shape) # torch.Size([270, 128, 21, 40])
        x = self.bn1(x)
        print(x.shape) # torch.Size([270, 128, 21, 40])
        x = self.relu(x)
        x = self.layer1(x)
        print(x.shape) # torch.Size([270, 128, 11, 20])
        exit()
        x = self.layer2(x)
        print(x.shape)
        x = self.layer3(x)
        print(x.shape)
        x = self.layer4(x)
        print(x.shape)
        x = x.squeeze(2)
        print(x.shape)
        return x

if __name__ == '__main__':

    x = torch.randn(torch.Size([10, 27, 60, 40]))
    attn_mask = torch.where(x == 0, -100000, 0)
    attn_mask = attn_mask[:, :, :, 0].squeeze(-1)

    # combine sentence-level and segment-level into batch dimensions
    x_resized = x.reshape(-1, 60, 40) # B, T, F
    attn_mask_resized = attn_mask.reshape(-1, 60) # B, T
    
    # CNN pooling 
    m = ResDavenet()
    m(x_resized)
    exit()
    

    # 1-layer attentive pooling  
    m = AttentivePooling(40, 40)
    y = m(x_resized, attn_mask_resized)
    y = y.reshape(10, 27, 40)
    print(y.shape)
