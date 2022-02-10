import torch 
import torch.nn as nn 
import math
import numpy as np

from torch.distributions import Categorical

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

def create_resdavenet_vq(args):
    vq_sizes = [int(s) for s in args.VQ_sizes.split(',')]
    layer_widths = [int(w) for w in args.layer_widths.split(',')]
    layer_depths = [int(w) for w in args.layer_depths.split(',')]
    vqs_enabled = [int(w) for w in args.VQ_turnon.split(',')]
    
    # Load Models
    if args.audio_model == 'ResDavenetVQ':
        audio_model = ResDavenetVQ(
                layers=layer_depths, 
                layer_widths=layer_widths, 
                convsize=args.convsize, 
                codebook_Ks=vq_sizes, 
                commitment_cost=args.VQ_commitment_cost, 
                jitter_p=args.jitter, 
                vqs_enabled=vqs_enabled, 
                init_ema_mass=args.init_ema_mass,
                init_std=args.init_std, 
                nonneg_init=args.nonneg_init)
    else:
        raise ValueError('Unknown audio model: %s' % audio_model_name)
    
    return audio_model

################ ResDaveNet-VQ below #####################

def conv1d(in_planes, out_planes, width=9, stride=1, bias=False):
    """1xd convolution with padding"""
    if width % 2 == 0:
        pad_amt = int(width / 2)
    else:
        pad_amt = int((width - 1) / 2)
    return nn.Conv2d(in_planes, out_planes, kernel_size=(1,width), 
                     stride=stride, padding=(0,pad_amt), bias=bias)

def flatten_tensor(inputs):
    """
    Convert a 4D tensor of shape (B, C, H, W) to a 2D tensor of shape 
    (B*H*W, C) and return (B, H, W, C) shape
    """
    inputs = inputs.permute(0, 2, 3, 1).contiguous()
    bhwc = inputs.shape
    return inputs.view(-1, bhwc[-1]), bhwc

def unflatten_tensor(inputs, bhwc):
    """
    Inverse function for flatten_tensor()
    """
    if inputs is None:
        return inputs
    return inputs.view(bhwc).permute(0, 3, 1, 2)

def get_flattened_indices(nframes, padded_len):
    indices = []
    for i, nframe in enumerate(nframes):
        indices.append(torch.arange(nframe) + i * padded_len)
    return torch.cat(indices).to(nframes.device)

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
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
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
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.squeeze(2)
        return x

class ResDavenetVQ(ResDavenet):
    def __init__(self, feat_dim=40, block=SpeechBasicBlock, 
                 layers=[2, 2, 2, 2], layer_widths=[128, 128, 256, 512, 1024],
                 convsize=9, codebook_Ks=[512, 512, 512, 512, 512], 
                 commitment_cost=1, jitter_p=0.0, vqs_enabled=[0, 0, 0, 0, 0], 
                 EMA_decay=0.99, init_ema_mass=1, init_std=1, 
                 nonneg_init=False):
        assert(len(codebook_Ks) == 5)
        assert(len(vqs_enabled) == 5)
        
        super().__init__(feat_dim=feat_dim, block=block, layers=layers, 
                         layer_widths=layer_widths, convsize=convsize)
        for l in range(5):
            if vqs_enabled[l]:
                quant_layer = VectorQuantizerEMA(
                        codebook_Ks[l], layer_widths[l], commitment_cost, 
                        decay=EMA_decay, init_ema_mass=init_ema_mass,
                        init_std=init_std, nonneg_init=nonneg_init)
                setattr(self, 'quant%d' % (l + 1), quant_layer)
        self.jitter_p = jitter_p
        self.jitter = TemporalJitter(p_left=jitter_p, p_right=jitter_p)
        self.vqs_enabled = list(vqs_enabled)

    def maybe_quantize(self, inputs, quant_idx, nframes=None):
        """
        Wrapper for quantization. Return flat_inputs and 
        flat_onehots for separate EMA codebook updates.
        
        Args:
            inputs (torch.Tensor): Pre-quantized inputs of shape (B, C, H, W).
            quant_idx (int): Index of the quantization layer to use.
            nframes (torch.Tensor): Lengths of shape (B,) w.r.t. inputs to the
                quantization layer (not the raw nframes to the model).
        Returns:
            flat_losses (torch.Tensor): Quantization loss for each frame. A
                tensor of shape (sum(nframes),)
            quant_inputs (torch.Tensor): Quantized input of shape (B, C, H, W).
            flat_inputs (torch.Tensor): Non-padding input frames. A tensor
                of shape (sum(nframes), C)
            flat_onehots (torch.Tensor): One-hot codes for non-padding input
                frames. A tensor of shape (sum(nframes), K)
        """
        flat_inputs, bhwc = flatten_tensor(inputs)
        ret_flat_inputs = flat_inputs
        if nframes is not None:
            indices = get_flattened_indices(nframes, bhwc[2]).long()
            indices = indices.to(inputs.device)
            ret_flat_inputs = torch.index_select(flat_inputs, 0, indices)

        if not self.vqs_enabled[quant_idx]:
            return None, inputs, ret_flat_inputs, None

        quant_layer = getattr(self, 'quant%d' % (quant_idx + 1))
        flat_losses, quant_inputs, flat_onehots = quant_layer(flat_inputs)
        quant_inputs = unflatten_tensor(quant_inputs, bhwc)
        if nframes is not None:
            flat_losses = torch.index_select(flat_losses, 0, indices)
            flat_onehots = torch.index_select(flat_onehots, 0, indices)

        return flat_losses, quant_inputs, ret_flat_inputs, flat_onehots

    def maybe_jitter(self, inputs):
        return self.jitter(inputs) if self.jitter_p > 0 else inputs

    def forward(self, x, nframes=None):
        """
        If nframes is provided, remove padded parts from quant_losses,
        flat_inputs and flat_onehots. This is useful for training, when EMA 
        only requires pre-quantized inputs and assigned indices. Note that
        jitter() is only applied after VQ-{2,3}.
        Args:
            x (torch.Tensor): Spectral feature batch of shape (B, C, F, T) or 
                (B, F, T).
            nframes (torch.Tensor): Number of frames for each utterance. Shape
                is (B,)
        """
        quant_losses = [None] * 5  # quantization losses by layer
        flat_inputs = [None] * 5   # flattened pre-quantized inputs by layer
        flat_onehots = [None] * 5  # flattened one-hot codes by layer
        
        if x.dim() == 3:
            x = x.unsqueeze(1)
        L = x.size(-1)
        cur_nframes = None

        x = self.relu(self.bn1(self.conv1(x)))
        if nframes is not None:
            cur_nframes = nframes / round(L / x.size(-1))
        (quant_losses[0], x, flat_inputs[0],
         flat_onehots[0]) = self.maybe_quantize(x, 0, cur_nframes)
        x = self.maybe_jitter(x)

        x = self.layer1(x)
        if nframes is not None:
            cur_nframes = nframes / round(L / x.size(-1))
        (quant_losses[1], x, flat_inputs[1],
         flat_onehots[1]) = self.maybe_quantize(x, 1, cur_nframes)
        x = self.maybe_jitter(x)
        
        x = self.layer2(x)
        if nframes is not None:
            cur_nframes = nframes / round(L / x.size(-1))
        (quant_losses[2], x, flat_inputs[2],
         flat_onehots[2]) = self.maybe_quantize(x, 2, cur_nframes)

        x = self.layer3(x)
        if nframes is not None:
            cur_nframes = nframes / round(L / x.size(-1))
        (quant_losses[3], x, flat_inputs[3],
         flat_onehots[3]) = self.maybe_quantize(x, 3, cur_nframes)
        
        x = self.layer4(x)
        if nframes is not None:
            cur_nframes = nframes / round(L / x.size(-1))
        (quant_losses[4], x, flat_inputs[4],
         flat_onehots[4]) = self.maybe_quantize(x, 4, cur_nframes)

        x = x.mean(-1).squeeze()

        return x, quant_losses, flat_inputs, flat_onehots

    def ema_update(self, inputs_by_layer, onehots_by_layer):
        """
        Exponential moving average update for enabled codebooks.
        Args:
            inputs_by_layer (list): A list of five torch.Tensor/None objects, 
                Each tensor is a pre-quantized input batch for a VQ layer.
                Shape is (N, D), where N is the number of frames, D is the 
                dimensionality of code embeddings.
            onehots_by_layer (list): A list of five torch.Tensor/None objects,
                which are onehot codes of shape (N, K), where K is the number
                of codes.
        """
        for quant_idx, is_enabled in enumerate(self.vqs_enabled):
            if not is_enabled:
                continue
            x, c = inputs_by_layer[quant_idx], onehots_by_layer[quant_idx]
            assert(x is not None)
            assert(c is not None)
            quant_layer = getattr(self, 'quant%d' % (quant_idx + 1))
            quant_layer.ema_update(x, c)

    def get_vq_outputs(self, x, layer, unflatten=False):
        """
        Get feature around the specified VQ layer. Jittering is not applied.
        """
        assert(layer in ['quant%d' % (d + 1) for d in range(5)])

        if x.dim() == 3:
            x = x.unsqueeze(1)
        L = x.size(-1)

        def _prepare_return(x, losses, preq_x, onehots):
            if unflatten:
                B, _, H, W = x.size()
                losses = unflatten_tensor(losses, (B, H, W, -1))
                preq_x = unflatten_tensor(preq_x, (B, H, W, -1))
                onehots = unflatten_tensor(onehots, (B, H, W, -1))
            return losses, x, preq_x, onehots

        x = self.relu(self.bn1(self.conv1(x)))
        losses, x, preq_x, onehots = self.maybe_quantize(x, 0)
        if layer == 'quant1':
            return _prepare_return(x, losses, preq_x, onehots)
        
        for quant_idx in range(1, 5):
            x = getattr(self, 'layer%d' % quant_idx)(x)
            losses, x, preq_x, onehots = self.maybe_quantize(x, quant_idx)
            if layer == 'quant%d' % (quant_idx + 1):
                return _prepare_return(x, losses, preq_x, onehots)

    def get_embedding(self, layer):
        """
        Get VQ embedding at the specified layer.
        """
        assert(hasattr(self, layer))
        return getattr(self, layer).get_embedding()

def compute_perplexity(onehots):
    avg_probs = torch.mean(onehots, dim=0)
    perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
    return perplexity

class TemporalJitter(nn.Module):
    def __init__(self, p_left=0.12, p_right=0.12):
        super(TemporalJitter, self).__init__()
        assert(p_left >= 0 and p_right >= 0)
        assert(p_left + p_right <= 1)
        self.p_left = p_left
        self.p_right = p_right
        self.p_middle = 1.0 - p_left - p_right
        self.sampler = Categorical(torch.tensor([p_left, self.p_middle, p_right])) 

    def forward(self, x):
        if self.training:
            jitters = self.sampler.sample(sample_shape=(x.size(-1),))
            Tinds = torch.arange(x.size(-1)) + jitters
            Tinds = Tinds.to(x.device).clamp(0, x.size(-1)-1)
            return torch.index_select(x, -1, Tinds)
        else:
            return x

class VectorQuantizerEMA(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost, decay, 
                 epsilon=1e-8, init_std=1, nonneg_init=False, init_ema_mass=1):
        super(VectorQuantizerEMA, self).__init__()
        
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        
        self._embedding = nn.Embedding(num_embeddings, embedding_dim)
        self._embedding.weight.requires_grad = False
        self._embedding.weight.data.normal_(std=init_std)
        if nonneg_init:
            self._embedding.weight.data.clamp_(min=0)
        self._commitment_cost = commitment_cost

        self.register_buffer(
                '_ema_cluster_size', init_ema_mass * torch.ones(num_embeddings))
        self._ema_w = nn.Parameter(
                init_ema_mass * torch.from_numpy(self._embedding.weight.cpu().numpy()),
                requires_grad=False)
        if nonneg_init:
            self._ema_w.data.clamp_(min=0)
        
        self._decay = decay
        self._epsilon = epsilon

    def forward(self, inputs):
        """
        Args:
            inputs (torch.Tensor): input of shape (B, D)
        Returns:
            loss (torch.Tensor): (B, D)
            quantized (torch.Tensor): (B, D)
            onehots (torch.Tensor): (B, num_embeddings)
        """
        # Calculate distances
        distances = (torch.sum(inputs**2, dim=1, keepdim=True) 
                    + torch.sum(self._embedding.weight**2, dim=1)
                    - 2 * torch.matmul(inputs, self._embedding.weight.t()))
            
        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        onehots = torch.zeros(encoding_indices.shape[0],
                              self._num_embeddings)
        onehots = onehots.to(inputs.device)
        onehots.scatter_(1, encoding_indices, 1)
        
        # Quantize and unflatten
        quantized = torch.matmul(onehots, self._embedding.weight)
        
        # Loss
        e_latent_loss = (quantized.detach() - inputs)**2
        loss = self._commitment_cost * e_latent_loss
        
        # Straight-through estimator for gradient
        if self.training:
            quantized = inputs + (quantized - inputs).detach()
        
        return loss, quantized, onehots

    def ema_update(self, inputs, onehots):
        """
        Args:
            inputs (torch.Tensor): input of shape (B, D)
            onehots (torch.Tensor): one-hot encodings of input of shape (B, K)
        """
        assert(self.training)

        # Use EMA to update the embedding vectors
        self._ema_cluster_size = self._ema_cluster_size * self._decay + \
                                 (1 - self._decay) * torch.sum(onehots, 0)

        # Redistribute cluster size by interpolating with a uniform assignment
        n = torch.sum(self._ema_cluster_size.data)
        self._ema_cluster_size = (
                (self._ema_cluster_size + self._epsilon)
                / (n + self._num_embeddings * self._epsilon) * n)
        
        dw = torch.matmul(onehots.t(), inputs)
        self._ema_w = nn.Parameter(
                self._ema_w * self._decay + (1 - self._decay) * dw, 
                requires_grad=False)
        
        self._embedding.weight = nn.Parameter(
                self._ema_w / self._ema_cluster_size.unsqueeze(1), 
                requires_grad=False)

    def get_embedding(self):
        return self._embedding.weight.detach()

if __name__ == '__main__':

    x = torch.randn(torch.Size([10, 27, 60, 40]))
    attn_mask = torch.where(x == 0, -100000, 0)
    attn_mask = attn_mask[:, :, :, 0].squeeze(-1)

    # combine sentence-level and segment-level into batch dimensions
    x_resized = x.reshape(-1, 60, 40) # B, T, F
    attn_mask_resized = attn_mask.reshape(-1, 60) # B, T
    
    # 1-layer attentive pooling  
    m = AttentivePooling(40, 40)
    y = m(x_resized, attn_mask_resized)
    y = y.reshape(10, 27, 40)
    print(y.shape)
