# c.f. fairseq/fairseq/models/wav2vec/wav2vec2.py, modified to return intermediate
# layer results

#
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import math
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List, Tuple

from fairseq import utils

from fairseq.models import BaseFairseqModel
from fairseq.modules import (
    Fp32GroupNorm,
    Fp32LayerNorm,
    GradMultiply,
    # GumbelVectorQuantizer,
    LayerNorm,
    MultiheadAttention,
    SamePad,
    TransposeLast,
)
from fairseq.modules.transformer_sentence_encoder import init_bert_params
from fairseq.utils import buffered_arange
# from fairseq.data.data_utils import compute_mask_indices, compute_mask_indices_correct
# from .gumbel_quantizer import GumbelVectorQuantizer, GumbelVectorQuantizer_correct

class Wav2Vec2Model_cls(BaseFairseqModel):
    @staticmethod
    def add_args(parser):
        parser.add_argument(
            "--extractor_mode",
            choices=["default", "layer_norm"],
            help="mode for feature extractor. default has a single group norm with d groups in the first conv block, whereas layer_norm has layer norms in every block (meant to use with --normalize)",
            default="default"
        )

        parser.add_argument(
            "--encoder_layers",
            type=int,
            metavar="L",
            help="num encoder layers in the transformer",
            default=12
        )
        parser.add_argument(
            "--encoder_embed_dim",
            type=int,
            metavar="H",
            help="encoder embedding dimension",
            default=768
        )
        parser.add_argument(
            "--encoder_ffn_embed_dim",
            type=int,
            metavar="F",
            help="encoder embedding dimension for FFN",
            default=3072
        )
        parser.add_argument(
            "--encoder_attention_heads",
            type=int,
            metavar="A",
            help="num encoder attention heads",
            default=12
        )
        parser.add_argument(
            "--activation-fn",
            choices=utils.get_available_activation_fns(),
            help="activation function to use",
            default="gelu"
        )

        parser.add_argument(
            "--dropout",
            type=float,
            metavar="D",
            help="dropout probability for the transformer",
            default=0.1
        )

        parser.add_argument(
            "--attention_dropout",
            type=float,
            metavar="D",
            help="dropout probability for attention weights",
            default=0.1
        )

        parser.add_argument(
            "--activation_dropout",
            type=float,
            metavar="D",
            help="dropout probability after activation in FFN",
            default=0.0
        )

        parser.add_argument(
            "--final_dim",
            type=int,
            metavar="D",
            help="project final representations and targets to this many dimensions",
            default=256 # 0 means use encoder_embed_dim as the final_dim
        )

        parser.add_argument(
            "--layer_norm_first",
            action="store_true",
            help="apply layernorm first in the transformer",
            default=False
        )

        parser.add_argument(
            "--encoder_layerdrop",
            type=float,
            help="probability of dropping a tarnsformer layer",
            default=0.0
        )

        parser.add_argument(
            "--conv_feature_layers",
            type=str,
            metavar="EXPR",
            help="convolutional feature extraction layers [(dim, kernel_size, stride), ...]",
            default= "[(512, 10, 5)] + [(512, 3, 2)] * 4 + [(512,2,2)] + [(512,2,2)]"
        )

        parser.add_argument(
            "--logit_temp", type=float, help="temperature to divide logits by", default=0.1
        )

        parser.add_argument(
            "--quantize_targets", action="store_true", help="use the original convnet feature as input to the transformer, quantized convnet features that are reconstruction targets when calculating the loss (w2v2 paper)", default=True
        )

        parser.add_argument(
            "--quantize_input", action="store_true", help="quantized all outputs of convnet feature extractor and use them as the input to the transformer", default=False
        )

        parser.add_argument(
            "--same_quantizer", action="store_true", help="use the same quantizer to quantize both transformer input and output", default=False
        )
        parser.add_argument(
            "--feature_grad_mult",
            type=float,
            help="multiply feature extractor var grads by this",
            default=0.1 # the paper use 0.1 i.e. scale down the gradient by a factor of 10
        )

        parser.add_argument(
            "--latent_vars",
            type=int,
            metavar="N",
            help="number of latent variables V in each group of the codebook",
            default=320
        )

        parser.add_argument(
            "--latent_groups",
            type=int,
            metavar="N",
            help="number of groups G of latent variables in the codebook",
            default=2
        )

        parser.add_argument(
            "--latent_dim",
            type=int,
            metavar="N",
            help="if > 0, uses this dimensionality for latent variables (code dimension). otherwise uses final_dim / latent_groups",
            default=0
        )

        parser.add_argument("--mask_length", type=int, help="mask length", default=10)

        parser.add_argument( "--mask_prob", type=float, help="probability of replacing a token with mask, default is 0.65, since mask_length=10, this is equivalent to all the tokens in a sequence has 0.065 probability to be starting tokens of a masked span", default=0.65)

        parser.add_argument(
            "--mask_selection",
            type=str,
            choices=["static", "uniform", "normal", "poisson"],
            help="how to choose masks",
            default="static"
        )

        parser.add_argument(
            "--mask_other",
            type=float,
            help="secondary mask argument (used for more complex distributions), see help in compute_mask_indices",
            default=0
        )

        parser.add_argument(
            "--no_mask_overlap",
            action="store_true",
            help="whether to allow masks to overlap",
            default=False
        )

        parser.add_argument(
            "--mask_min_space",
            type=int,
            help="min space between spans (if no overlap is enabled)",
            default=1
        )

        parser.add_argument(
            "--mask_channel_length",
            type=int,
            help="repeat the mask indices multiple times",
            default=0.0
        )

        parser.add_argument(
            "--mask_channel_prob",
            type=float,
            help="probability of replacing a token with mask",
            default=0.0
        )

        parser.add_argument(
            "--mask_channel_selection",
            type=str,
            choices=["static", "uniform", "normal", "poisson"],
            help="how to choose masks",
            default="static"
        )

        parser.add_argument(
            "--mask_channel_other",
            type=float,
            help="secondary mask argument (used for more complex distributions), see help in compute_mask_indices",
            default=0
        )

        parser.add_argument(
            "--no_mask_channel_overlap",
            action="store_true",
            help="whether to allow masks to overlap",
            default=False
        )

        parser.add_argument(
            "--mask_channel_min_space",
            type=int,
            help="min space between spans (if no overlap is enabled)",
            default=1
        )

        parser.add_argument(
            "--dropout_input",
            type=float,
            metavar="D",
            help="dropout to apply to the input (after feat extr)",
            default=0.1
        )

        parser.add_argument(
            "--dropout_features",
            type=float,
            metavar="D",
            help="dropout to apply to the features (after feat extr)",
            default=0.1
        )

        parser.add_argument(
            "--num_negatives", type=int, metavar="N", help="number of negative examples",
            default=100
        )

        parser.add_argument(
            "--negatives_from_everywhere",
            action="store_true",
            help="sample negatives from everywhere, not just masked states",
            default=False
        )

        parser.add_argument(
            "--cross_sample_negatives",
            type=int,
            metavar="N",
            help="num of cross sampled negatives",
            default=0
        )

        parser.add_argument(
            "--codebook_negatives",
            type=int,
            metavar="N",
            help="num of codebook sampled negatives",
            default=0
        )

        parser.add_argument(
            "--conv_pos",
            type=int,
            metavar="N",
            help="kernel size for convolutional positional embeddings",
            default=128
        )

        parser.add_argument(
            "--conv_pos_groups",
            type=int,
            metavar="N",
            help="number of groups for convolutional positional embedding",
            default=16
        )

        parser.add_argument(
            "--latent_temp",
            type=str,
            metavar="D",
            help="temperature for latent variable sampling. can be string of tuple of 3 values (start, end, decay)",
            default="(2,0.5,0.999995)"
        )

        parser.add_argument(
            "--target_glu", action="store_true", help="adds projection + glu to targets", default=False
        )

        parser.add_argument(
            "--conv-bias", action="store_true", help="include bias in conv encoder", default=False
        )

        parser.add_argument(
            "--layer_use", type=int, help="which layer feat to use to input to second tranformer, range from 0 to encoder_layer - 1", default=4
        )
        
        parser.add_argument(
            "--diversity_weight", type=float, help="weight on the diversity loss", default=0.1
        )

        parser.add_argument(
            "--return_code_index", action="store_true", default=False, help="return the code index"
        )
        
        
        parser.add_argument(
            "--trim_mask", action="store_true", default=False
        )
        
        # hubert args
        parser.add_argument("--untie_final_proj", type = bool, default=True,
            help = "use separate projection for each target"
        )
        parser.add_argument("--skip_masked", type = bool, default=False,
            help = "skip computing losses over masked frames"
        )
        parser.add_argument("--skip_nomask", type = bool, default=False,
            help = "skip computing losses over unmasked frames"
        )
        parser.add_argument(
            "--pred_masked_weight", type=float, default=1.0
        )
        parser.add_argument(
            "--pred_nomask_weight", type=float, default=0.0
        )

        # feature_extractor2 and encoder2
        parser.add_argument(
            "--conv2_feature_layers",
            type=str,
            metavar="EXPR",
            help="convolutional feature extraction layers [(dim, kernel_size, stride), ...]",
            default= "[(768, 3, 1)] + [(512, 3, 2)] + [(512, 3, 2)]" # T: 899 -> 223, receptive field: __ frames, i.e. __ ms (but not really)
        )
        
        parser.add_argument(
            "--trm2_layers", type=int, default=None, help="number of layers of second encoder"
        )
        
        parser.add_argument(
            "--conv2_pos", type=int,
            help="kernel size for convolutional positional embeddings after feature_extractor2, if passing None, means do not use positional embedding in encoder2. If use, 32 is a good number, as 32 = 128 / 4",
            default=None
        )

        parser.add_argument(
            "--trm2_layer_use", type=int, help="which layers in the encoder2 to use", default=None
        )

        # whether or not using cls token
        parser.add_argument(
            "--use_audio_cls_token", action="store_true", default=False
        )

        # use level2 audio loss
        parser.add_argument(
            "--level2_audio_loss", action="store_true", default=False
        )

        parser.add_argument(
            "--level2_mask_prob_multi", type=float, default=1.
        )
        
        parser.add_argument(
            "--level2_mask_len_multi", type=float, default=0.2
        )

        parser.add_argument(
            "--random_init_last_x", type=int, default=None
        )
        
        parser.add_argument(
            "--freeze_first_x", type=int, default=None
        )
    def __init__(self, args):
        super().__init__()
        self.args = args
        # if self.args.trim_mask:
        #     self.Compute_Mask_Indices = compute_mask_indices
        #     Quantizer = GumbelVectorQuantizer
        # else:
        #     self.Compute_Mask_Indices = compute_mask_indices_correct
        #     Quantizer = GumbelVectorQuantizer_correct
        self.Compute_Mask_Indices = None
        Quantizer = None
        feature_enc_layers = eval(args.conv_feature_layers)
        self.embed = feature_enc_layers[-1][0]

        self.feature_extractor = ConvFeatureExtractionModel(
            conv_layers=feature_enc_layers,
            dropout=0.0,
            mode=args.extractor_mode,
            conv_bias=args.conv_bias,
        )

        self.post_extract_proj = (
            nn.Linear(self.embed, args.encoder_embed_dim)
            if self.embed != args.encoder_embed_dim and not args.quantize_input
            else None
        )

        self.mask_prob = args.mask_prob
        self.mask_selection = args.mask_selection
        self.mask_other = args.mask_other
        self.mask_length = args.mask_length
        self.no_mask_overlap = args.no_mask_overlap
        self.mask_min_space = args.mask_min_space

        self.mask_channel_prob = args.mask_channel_prob
        self.mask_channel_selection = args.mask_channel_selection
        self.mask_channel_other = args.mask_channel_other
        self.mask_channel_length = args.mask_channel_length
        self.no_mask_channel_overlap = args.no_mask_channel_overlap
        self.mask_channel_min_space = args.mask_channel_min_space

        self.dropout_input = nn.Dropout(args.dropout_input)
        self.dropout_features = nn.Dropout(args.dropout_features)

        self.feature_grad_mult = args.feature_grad_mult

        self.quantizer = None
        self.input_quantizer = None

        self.n_negatives = args.num_negatives
        self.cross_sample_negatives = args.cross_sample_negatives
        self.codebook_negatives = args.codebook_negatives
        self.negatives_from_everywhere = args.negatives_from_everywhere

        self.logit_temp = args.logit_temp

        self.diversity_weight = args.diversity_weight
        
        if self.args.caption_w2v2_weight == None and self.args.libri_w2v2_weight == None and self.args.caption_hubert_weight == None and self.args.libri_hubert_weight == None:
            self.w2v2_loss = False
        else:
            self.w2v2_loss = True
        if args.quantize_input and self.w2v2_loss:
            vq_dim = args.latent_dim if args.latent_dim > 0 else args.encoder_embed_dim
            self.input_quantizer = (
                Quantizer(
                    dim=self.embed,
                    num_vars=args.latent_vars,
                    temp=eval(args.latent_temp),
                    groups=args.latent_groups,
                    combine_groups=False,
                    vq_dim=vq_dim,
                    time_first=True,
                )
                if not args.same_quantizer
                else self.quantizer
            )
            self.project_inp = nn.Linear(vq_dim, args.encoder_embed_dim)

        final_dim = args.final_dim if args.final_dim > 0 else args.encoder_embed_dim

        if args.quantize_targets and self.w2v2_loss:
            vq_dim = args.latent_dim if args.latent_dim > 0 else final_dim
            self.quantizer = Quantizer(
                dim=self.embed,
                num_vars=args.latent_vars,
                temp=eval(args.latent_temp),
                groups=args.latent_groups,
                combine_groups=False,
                vq_dim=vq_dim,
                time_first=True,
            )
            self.project_q = nn.Linear(vq_dim, final_dim)
        else:
            self.project_q = nn.Linear(self.embed, final_dim)

        self.mask_emb = nn.Parameter(
            torch.FloatTensor(args.encoder_embed_dim).uniform_()
        )
        self.pos_conv = nn.Conv1d(
                self.args.encoder_embed_dim,
                self.args.encoder_embed_dim,
                kernel_size=args.conv_pos,
                padding=args.conv_pos // 2,
                groups=args.conv_pos_groups,
            )
        dropout = 0
        std = math.sqrt((4 * (1.0 - dropout)) / (args.conv_pos * self.args.encoder_embed_dim))
        nn.init.normal_(self.pos_conv.weight, mean=0, std=std)
        nn.init.constant_(self.pos_conv.bias, 0)

        self.pos_conv = nn.utils.weight_norm(self.pos_conv, name="weight", dim=2)
        self.pos_conv = nn.Sequential(self.pos_conv, SamePad(args.conv_pos), nn.GELU())
        if not self.w2v2_loss:
            args.encoder_layers = args.layer_use + 1
        self.encoder = TransformerEncoder(args)
        self.layer_norm = LayerNorm(self.embed)

        self.target_glu = None
        if args.target_glu:
            self.target_glu = nn.Sequential(
                nn.Linear(final_dim, final_dim * 2), nn.GLU()
            )

        self.final_proj = nn.Linear(args.encoder_embed_dim, final_dim)

        if self.args.use_audio_cls_token:
            self.cls_token = torch.nn.Parameter(torch.randn((1, 1, args.encoder_embed_dim)), requires_grad=True)
            if self.args.trm2_layers != None:
                self.cls_token_proj = nn.Linear(args.encoder_embed_dim, args.encoder_embed_dim)
        
        if self.args.trm2_layers != None:
            feature_enc_layers2 = eval(args.conv2_feature_layers)
            self.feature_extractor2 = ConvFeatureExtractionModel(
                conv_layers=feature_enc_layers2,
                dropout=0.1,
                mode=args.extractor_mode,
                conv_bias=args.conv_bias,
                first_conv=False
            )
            embed2 = feature_enc_layers2[-1][0]
            self.layer_norm2 = LayerNorm(embed2)
            self.dropout_input2 = nn.Dropout(args.dropout_input)
            self.dropout_features2 = nn.Dropout(args.dropout_features)
            self.post_extract_proj2 = nn.Linear(embed2, args.encoder_embed_dim) # dim of self.feature_extractor2 output is still 512, so need proj to 768
            # self.dropout_features2 = nn.Dropout(args.dropout_features)
            if args.conv2_pos != None:
                self.pos_conv2 = nn.Conv1d(
                        self.args.encoder_embed_dim,
                        self.args.encoder_embed_dim,
                        kernel_size=args.conv2_pos,
                        padding=args.conv2_pos // 2,
                        groups=args.conv_pos_groups,
                    )
                dropout = 0
                std = math.sqrt((4 * (1.0 - dropout)) / (args.conv_pos * self.args.encoder_embed_dim))
                nn.init.normal_(self.pos_conv2.weight, mean=0, std=std)
                nn.init.constant_(self.pos_conv2.bias, 0)

                self.pos_conv2 = nn.utils.weight_norm(self.pos_conv2, name="weight", dim=2)
                self.pos_conv2 = nn.Sequential(self.pos_conv2, SamePad(args.conv_pos), nn.GELU())
            trm2_args = argparse.Namespace(**vars(args))
            # trm2_args = args
            trm2_args.encoder_layers = args.trm2_layers
            # trm2_args.layer_use = args.trm2_layer_use
            trm2_args.layer_use = args.trm2_layers - 1  if args.trm2_layer_use is None else args.trm2_layer_use # this mean we use the last layer's output, i.e. ideally don't consider audio constrastive loss at this level
            self.encoder2 = TransformerEncoder(trm2_args)
            if self.args.level2_audio_loss and self.w2v2_loss:
                self.mask_emb2 = nn.Parameter(
                    torch.FloatTensor(args.encoder_embed_dim).uniform_()
                )
                self.quantizer2 = Quantizer(
                    dim=self.embed,
                    num_vars=args.latent_vars//2, # 320//2 = 160 use less codes intentionally
                    temp=eval(args.latent_temp),
                    groups=args.latent_groups,
                    combine_groups=False,
                    vq_dim=args.latent_dim if args.latent_dim > 0 else final_dim,
                    time_first=True,
                )
                self.project_q2 = nn.Linear(vq_dim, final_dim)
                self.target_glu2 = None
                if args.target_glu:
                    self.target_glu2 = nn.Sequential(
                        nn.Linear(final_dim, final_dim * 2), nn.GLU()
                    )
                self.final_proj2 = nn.Linear(args.encoder_embed_dim, final_dim)


    def apply_mask(self, x, padding_mask):
        B, T, C = x.shape
        if self.mask_prob > 0:
            mask_indices = self.Compute_Mask_Indices(
                (B, T),
                padding_mask,
                self.mask_prob,
                self.mask_length,
                self.mask_selection,
                self.mask_other,
                min_masks=2,
                no_overlap=self.no_mask_overlap,
                min_space=self.mask_min_space,
            )
            mask_indices = torch.from_numpy(mask_indices).to(x.device)
            x[mask_indices] = self.mask_emb.to(x) # to x to avoid fp16 problem
        else:
            mask_indices = None

        if self.mask_channel_prob > 0:
            mask_channel_indices = self.Compute_Mask_Indices(
                (B, C),
                None,
                self.mask_channel_prob,
                self.mask_channel_length,
                self.mask_channel_selection,
                self.mask_channel_other,
                no_overlap=self.no_mask_channel_overlap,
                min_space=self.mask_channel_min_space,
            )
            mask_channel_indices = (
                torch.from_numpy(mask_channel_indices)
                .to(x.device)
                .unsqueeze(1)
                .expand(-1, T, -1)
            )
            x[mask_channel_indices] = 0

        return x, mask_indices
    
    def apply_mask2(self, x, padding_mask):
        B, T, C = x.shape
        if self.mask_prob > 0:
            mask_indices = self.Compute_Mask_Indices(
                (B, T),
                padding_mask,
                self.mask_prob*self.args.level2_mask_prob_multi, # default 0.5 * 1. = 0.5
                int(self.mask_length*self.args.level2_mask_len_multi), # default 10 * 0.2 = 2
                self.mask_selection,
                self.mask_other,
                min_masks=1,
                no_overlap=self.no_mask_overlap,
                min_space=self.mask_min_space,
            )
            mask_indices = torch.from_numpy(mask_indices).to(x.device)
            x[mask_indices] = self.mask_emb2.to(x) # to x to avoid fp16 problem
        else:
            mask_indices = None

        if self.mask_channel_prob > 0:
            mask_channel_indices = self.Compute_Mask_Indices(
                (B, C),
                None,
                self.mask_channel_prob,
                self.mask_channel_length,
                self.mask_channel_selection,
                self.mask_channel_other,
                no_overlap=self.no_mask_channel_overlap,
                min_space=self.mask_channel_min_space,
            )
            mask_channel_indices = (
                torch.from_numpy(mask_channel_indices)
                .to(x.device)
                .unsqueeze(1)
                .expand(-1, T, -1)
            )
            x[mask_channel_indices] = 0

        return x, mask_indices
    
    

    def sample_negatives_trim_mask(self, y, num):

        if self.n_negatives == 0 and self.cross_sample_negatives == 0:
            return y.new(0)

        bsz, tsz, fsz = y.shape
        y = y.view(-1, fsz)  # BTC => (BxT)C

        cross_high = tsz * bsz
        high = tsz
        with torch.no_grad():
            assert high > 1, f"{bsz,tsz,fsz}"

            if self.n_negatives > 0:
                tszs = (
                    buffered_arange(num)
                    .unsqueeze(-1)
                    .expand(-1, self.n_negatives)
                    .flatten()
                )

                neg_idxs = torch.randint(
                    low=0, high=high - 1, size=(bsz, self.n_negatives * num)
                )
                neg_idxs[neg_idxs >= tszs] += 1

            if self.cross_sample_negatives > 0:
                tszs = (
                    buffered_arange(num)
                    .unsqueeze(-1)
                    .expand(-1, self.cross_sample_negatives)
                    .flatten()
                )

                cross_neg_idxs = torch.randint(
                    low=0,
                    high=cross_high - 1,
                    size=(bsz, self.cross_sample_negatives * num),
                )
                cross_neg_idxs[cross_neg_idxs >= tszs] += 1

        if self.n_negatives > 0:
            for i in range(1, bsz):
                neg_idxs[i] += i * high
        else:
            neg_idxs = cross_neg_idxs

        if self.cross_sample_negatives > 0 and self.n_negatives > 0:
            neg_idxs = torch.cat([neg_idxs, cross_neg_idxs], dim=1)

        negs = y[neg_idxs.view(-1)]
        negs = negs.view(
            bsz, num, self.n_negatives + self.cross_sample_negatives, fsz
        ).permute(
            2, 0, 1, 3
        )  # to NxBxTxC
        return negs, neg_idxs

    def sample_negatives(self, y, mask_indices):
        # num is y.shape[1], i.e. T_pred
        # now it's different for each instance
        if self.n_negatives == 0 and self.cross_sample_negatives == 0:
            return y.new(0)

        bt, fsz = y.shape
        T_preds = mask_indices.sum(1).cpu()
        cross_high = bt
        # high = tsz
        with torch.no_grad():

            if self.n_negatives > 0:
                neg_idxs = []
                for i, T_pred in enumerate(T_preds):
                    assert T_pred > 1, f"{T_pred}"
                    neg_idx = torch.randint(
                        low=0, high=T_pred - 1, size=(1,self.n_negatives * T_pred)
                    ).squeeze()
                    tszs = (
                        buffered_arange(T_pred)
                        .unsqueeze(-1)
                        .expand(-1, self.n_negatives)
                        .flatten()
                    )
                    neg_idx[neg_idx >= tszs] += 1
                    neg_idxs.append(neg_idx)

            if self.cross_sample_negatives > 0:
                cross_neg_idxs = []
                for i, T_pred in enumerate(T_preds):
                    assert T_pred > 1, f"{T_pred}"
                    cross_neg_idx = torch.randint(
                        low=0, high=cross_high - 1, size=(self.n_negatives * T_pred,)
                    )
                    tszs = (
                        buffered_arange(T_pred)
                        .unsqueeze(-1)
                        .expand(-1, self.cross_sample_negatives)
                        .flatten()
                    )
                    cross_neg_idx[cross_neg_idx >= tszs] += 1
                    cross_neg_idxs.append(cross_neg_idx)

        if self.n_negatives > 0:
            accum = 0
            for i in range(1, len(T_preds)):
                accum += T_preds[i-1]
                # print(neg_idx)
                # print("\n")
                # print(T_preds[i-1])
                
                neg_idxs[i] += accum
        else:
            neg_idxs = cross_neg_idxs

        if self.cross_sample_negatives > 0 and self.n_negatives > 0:
            neg_idxs = torch.cat([neg_idxs, cross_neg_idxs], dim=1)
        neg_idxs = torch.cat(neg_idxs).view(-1)
        negs = y[neg_idxs] # [T_pred1*n_negatives+T_pred2*n_negatives+...+T_predB*n_negatives, fsz]
        # negs = negs.view(
        #     bsz, num, self.n_negatives + self.cross_sample_negatives, fsz
        # ).permute(
        #     2, 0, 1, 3
        # )  # to NxBxTxC
        return negs, neg_idxs

    def compute_preds_trim_mask(self, x, y, negatives):
        # negatives: NxBxTxC
        # y is target: BxTxC
        # x is prediction: BxTxC
        neg_is_pos = (y == negatives).all(-1)
        y = y.unsqueeze(0)
        targets = torch.cat([y, negatives], dim=0) # (N+1)xBxTxC

        logits = torch.cosine_similarity(x.float(), targets.float(), dim=-1).type_as(x) # this gives [N+1, B, T]

        logits /= self.logit_temp

        if neg_is_pos.any():
            logits[1:][neg_is_pos] = float("-inf")

        return logits

    def compute_preds(self, x, y, negatives, mask_indices):
        # negatives: [T_pred1*negs+T_pred2*negs+...+T_predB*negs, D] 
        # y is target: [T_pred1+T_pred2+...+T_predB, D] 
        # x is prediction: [T_pred1+T_pred2+...+T_predB, D]
        accum = 0
        all_logits = []
        for i, mask_idx in enumerate(mask_indices):
            T_pred = mask_idx.long().sum()
            cur_x = x[accum:accum+T_pred, :]
            cur_y = y[accum:accum+T_pred, :]
            cur_negatives = negatives[accum*self.n_negatives:(accum+T_pred)*self.n_negatives, :].view(T_pred, self.n_negatives, -1).transpose(0,1)
            neg_is_pos = (cur_y == cur_negatives).all(-1)
            cur_y = cur_y.unsqueeze(0)
            targets = torch.cat([cur_y, cur_negatives], dim=0) # (N+1)xT_predxC

            logits = torch.cosine_similarity(cur_x.float(), targets.float(), dim=-1).type_as(cur_x) # this gives [N+1, T_pred]

            logits /= self.logit_temp

            if neg_is_pos.any():
                logits[1:][neg_is_pos] = float("-inf")
            all_logits.append(logits)
        return torch.cat(all_logits,dim=-1) # [N+1, T_pred1+T_pred2+...+T_predB]

    def forward(self, source, padding_mask=None, mask=True, features_only=False, superb=False, segment_layer=None, feature_layer=None, need_attention_weights=False, level2=False):
        # use tgt_layer if you want encoder to stop forward passing at that layer, usually used in testing retrieval accuracy

        if self.feature_grad_mult > 0:
            features = self.feature_extractor(source)
            if self.feature_grad_mult != 1.0:
                features = GradMultiply.apply(features, self.feature_grad_mult)
        else:
            with torch.no_grad():
                features = self.feature_extractor(source)

        features_pen = features.float().pow(2).mean()

        features = features.transpose(1, 2)
        features = self.layer_norm(features)
        unmasked_features = features.clone()

        if padding_mask is not None: # resize padding mask according to Conv pooling ratio
            extra = padding_mask.size(1) % features.size(1)
            if extra > 0:
                padding_mask = padding_mask[:, :-extra]
            padding_mask = padding_mask.view(padding_mask.size(0), features.size(1), -1)
            padding_mask = padding_mask.all(-1)
        if self.post_extract_proj is not None:
            features = self.post_extract_proj(features)
        if superb:
            all_feats_wo_cls = [features]

        features = self.dropout_input(features)
        unmasked_features = self.dropout_features(unmasked_features)

        num_vars = None
        code_ppl = None
        prob_ppl = None
        curr_temp = None
        
        # this have never being used in w2v2, the inputs are not quantized
        if self.input_quantizer:
            q = self.input_quantizer(features, produce_targets=False)
            features = q["x"]
            num_vars = q["num_vars"]
            code_ppl = q["code_perplexity"]
            prob_ppl = q["prob_perplexity"]
            curr_temp = q["temp"]
            features = self.project_inp(features)

        if mask:
            x, mask_indices = self.apply_mask(features, padding_mask) # x will be masked, note that mask is some learned embedding (768 dim), not zero!
            if mask_indices is not None:
                if self.args.trim_mask:
                    y = unmasked_features[mask_indices].view(unmasked_features.size(0), -1, unmasked_features.size(-1))
                else:
                    y = unmasked_features[mask_indices].view(-1, unmasked_features.size(-1))
            else:
                y = unmasked_features
            # print(x.shape)
            # print(f"num of tokens {(~padding_mask).sum(1).sum()}")
            # print(f"num of masks {mask_indices.sum(1).sum()}")
            # print(f"Level1 percentage of maksed tokens: {float(mask_indices.sum(1).sum()*100/(~padding_mask).sum(1).sum())}")
        else:
            x = features
            y = unmasked_features
            mask_indices = None
        # print(f"audio length at level1: {(~padding_mask).sum(1)}")
        # print(f"num of masked indices at level1 {mask_indices.sum(1)}")
        x_conv = self.pos_conv(x.transpose(1, 2))
        x_conv = x_conv.transpose(1, 2)
        x += x_conv
        # ############################################
        if self.args.use_audio_cls_token:
            x = torch.cat([self.cls_token.repeat(x.shape[0],1,1), x], dim=1)
            if padding_mask != None:
                cls_token_padding_mask = torch.zeros((padding_mask.shape[0],1)).to(padding_mask)
                padding_mask = torch.cat([cls_token_padding_mask, padding_mask], dim=1)
        # ############################################
        if superb:
            all_feats = self.encoder(x, padding_mask=padding_mask, superb=True)
            for feat in all_feats:
                all_feats_wo_cls.append(feat[:,1:])
            return {"last_hidden_state": all_feats_wo_cls[-1], "hidden_states": all_feats_wo_cls}

        if need_attention_weights:
            layer_features, attn_weights = self.encoder(x, padding_mask=padding_mask, segment_layer=segment_layer, feature_layer=feature_layer, need_attention_weights=True)
            return {"layer_features": layer_features, "attn_weights": attn_weights, "padding_mask": padding_mask} # layer_features is a list of features from specified feature_layer, each entry is a tensor of size [bsz, T+1, D]; attn_weights is a tensor of size [bsz, num_heads, T+1, T+1]

        features, layer_feats = self.encoder(x, padding_mask=padding_mask, tgt_layer=tgt_layer)
        # if self.args.double_cls:
        if self.args.use_audio_cls_token:
            cls_token = layer_feats[:,0]
            features = features[:,1:]
            layer_feats = layer_feats[:,1:]
            if padding_mask != None:
                padding_mask = padding_mask[:,1:]
        else:
            cls_token = None
        

        if self.quantizer and mask:
            q = self.quantizer(y, produce_targets=False, mask_indices=None)
            y = q["x"] # always only contains masked target
            num_vars = q["num_vars"]
            code_ppl = q["code_perplexity"]
            prob_ppl = q["prob_perplexity"]
            curr_temp = q["temp"]
            y = self.project_q(y)
            # print(y.shape)

            if self.negatives_from_everywhere:
                raise NotImplementedError
            else:
                if self.args.trim_mask:
                    negs, _ = self.sample_negatives_trim_mask(y, y.size(1))
                else:
                    negs, _ = self.sample_negatives(y, mask_indices) # negs: [T_pred1+T_pred2+...+T_predB, D] 

            if self.codebook_negatives > 0:
                raise NotImplementedError
        else:
            negs = None


        result = {"features": features, "layer_feats": layer_feats, "cls_token": cls_token, "padding_mask": padding_mask, "features_pen": features_pen, "negs": negs, "masked_target": y, "mask_indices": mask_indices}
        # print("955", layer_feats.dtype)
        if prob_ppl is not None:
            result["prob_perplexity"] = prob_ppl
            result["code_perplexity"] = code_ppl
            result["num_vars"] = num_vars
            result["temp"] = curr_temp
        if self.args.trm2_layers != None:
            ######################## TRM2 ########################
            x = layer_feats.transpose(1,2)
            x = self.feature_extractor2(x)
            x = x.transpose(1,2)
            x = self.layer_norm2(x)
            if padding_mask is not None: # resize padding mask according to Conv2 pooling ratio
                extra = padding_mask.size(1) % x.size(1)
                if extra > 0:
                    padding_mask = padding_mask[:, :-extra]
                padding_mask = padding_mask.view(padding_mask.size(0), x.size(1), -1)
                padding_mask = padding_mask.all(-1)
            if self.args.level2_audio_loss and self.w2v2_loss and mask:
                features = x
                unmasked_features = self.dropout_features2(x.clone())
                features = self.post_extract_proj2(features)
                features = self.dropout_input2(features)
                x, mask_indices = self.apply_mask2(features, padding_mask) # x will be masked, note that mask is some learned embedding (dim = 768), not zero!
                # print(f"audio length at level2: {(~padding_mask).sum(1)}")
                # print(f"num of masked indices at level2 {mask_indices.sum(1)}")
                # print(f"Level2 percentage of maksed tokens: {float(mask_indices.sum(1).sum()*100/(~padding_mask).sum(1).sum())}")
                assert mask_indices is not None
                if self.args.trim_mask:
                    y = unmasked_features[mask_indices].view(unmasked_features.size(0), -1, unmasked_features.size(-1))
                else:
                    y = unmasked_features[mask_indices].view(-1, unmasked_features.size(-1))
                q = self.quantizer2(y, produce_targets=False, mask_indices=None)
                y = q["x"] # always only contains masked target
                num_vars = q["num_vars"]
                code_ppl = q["code_perplexity"]
                prob_ppl = q["prob_perplexity"]
                curr_temp = q["temp"]
                y = self.project_q2(y)
                # print(y.shape)

                if self.args.trim_mask:
                    negs, _ = self.sample_negatives_trim_mask(y, y.size(1))
                else:
                    negs, _ = self.sample_negatives(y, mask_indices) # negs: [T_pred1+T_pred2+...+T_predB, D] 
                result["prob_perplexity2"] = prob_ppl
                result["code_perplexity2"] = code_ppl
                result["num_vars2"] = num_vars
                result["temp2"] = curr_temp
                result.update({"negs2": negs, "masked_target2": y, "mask_indices2": mask_indices})
            else:
                x = self.post_extract_proj2(x)
                x = self.dropout_input2(x)

            # print("1008", x.dtype)
            if self.args.conv2_pos is not None:
                x_conv = self.pos_conv2(x.transpose(1, 2))
                x_conv = x_conv.transpose(1, 2)
                x += x_conv

            if self.args.use_audio_cls_token:
                cls_token = self.cls_token_proj(cls_token).unsqueeze(1)
                x = torch.cat([cls_token, x], dim=1)
                if padding_mask != None:
                    cls_token_padding_mask = torch.zeros((padding_mask.shape[0],1)).to(padding_mask)
                    padding_mask = torch.cat([cls_token_padding_mask, padding_mask], dim=1)
            # print("1020", x.dtype)
            if level2:
                features, attn_weights = self.encoder2(x, padding_mask=padding_mask, tgt_layer=tgt_layer, need_attention_weights=True, pre_feats=pre_feats)
                return {"features": features, "attn_weights": attn_weights, "padding_mask": padding_mask}
            features, layer_feats = self.encoder2(x, padding_mask=padding_mask)
            if self.args.use_audio_cls_token:
                cls_token = layer_feats[:,0]
                features = features[:,1:]
                layer_feats = layer_feats[:,1:]
                if padding_mask != None:
                    padding_mask = padding_mask[:,1:]
            else:
                cls_token = None

            result.update({"cls_token": cls_token, "layer_feats": layer_feats, "padding_mask": padding_mask, "features2": features})

        return result

    def quantize(self, x):
        assert self.quantizer is not None
        x = self.feature_extractor(x)
        x = x.transpose(1, 2)
        x = self.layer_norm(x)
        return self.quantizer.forward_idx(x)

    def extract_features(self, source, padding_mask, mask=False):
        res = self.forward(source, padding_mask, mask=mask, features_only=True)
        return res["x"], res["padding_mask"]

    def get_logits(self, net_output):
        logits = net_output["x"]
        logits = logits.transpose(0, 2)
        logits = logits.reshape(-1, logits.size(-1))
        return logits

    def get_targets(self, sample, net_output, expand_steps=True):
        x = net_output["x"]
        return x.new_zeros(x.size(1) * x.size(2), dtype=torch.long)

    def get_extra_losses(self, net_output):
        pen = []

        if "prob_perplexity" in net_output:
            temp = (net_output["num_vars"] - net_output["prob_perplexity"]) / net_output["num_vars"]
            pen.append(temp*self.diversity_weight)

        if "features_pen" in net_output and self.feature_grad_mult > 0:
            pen.append(net_output["features_pen"])

        return pen
    
    def get_extra_losses2(self, net_output):
        pen = []

        if "prob_perplexity2" in net_output:
            temp = (net_output["num_vars2"] - net_output["prob_perplexity2"]) / net_output["num_vars"]
            pen.append(temp*self.diversity_weight)

        # if "features_pen2" in net_output and self.feature_grad_mult > 0:
        #     pen.append(net_output["features_pen2"])

        return pen

    def remove_pretraining_modules(self):
        self.quantizer = None
        self.project_q = None
        self.target_glu = None
        self.final_proj = None
        
    def carefully_load_state_dict(self, states, load_all=False):
        """
        1) Take care of DataParallel/nn.Module state_dict
        2) Show keys that are not loaded due to size mismatch or not found in model
        """
        random_init_last_x = getattr(self.args, "random_init_last_x", None)
        freeze_first_x = getattr(self.args, "freeze_first_x", None)
        if random_init_last_x != None and not load_all:
            cut = self.args.encoder_layers - random_init_last_x
            assert cut >= 0
            random_init_names = [f'encoder.layers.{i}.' for i in range(cut, self.args.encoder_layers)]
            print(f"randomly reinitialize the weights start with the following: {random_init_names}\n")
        new_states = self.state_dict()
        loaded_keys = []
        for k, v in states.items():
            k = k[7:] if k.startswith('module') else k
            k = k[22:] if k.startswith('w2v_encoder.w2v_model') else k
            k = k[11:] if k.startswith('w2v2_model') else k
            k = k[8:] if k.startswith('encoder.pos_conv') else k
            k = k[22:] if k.startswith('conv1_trm1_conv2_trm2') else k
            if random_init_last_x != None and not load_all:
                for names in random_init_names:
                    if k.startswith(names):
                        v = torch.tensor([0.0]).to(v.device) # make it so that the size doesn't match
                        break
            if k in new_states and new_states[k].size() == v.size():
                new_states[k] = v
                loaded_keys.append(k)
            else:
                print('Ignoring %s due to not existing or size mismatch' % k)

        non_loaded_keys = set(new_states.keys()) - set(loaded_keys)
        if non_loaded_keys:
            print('\nModel states that do not exist in the seed_dir:')
            for k in sorted(non_loaded_keys):
                print('  %s' % k)
        
        self.load_state_dict(new_states)
        print("")
        if freeze_first_x != None:
            freeze_names =  [f'encoder.layers.{i}.' for i in range(freeze_first_x)]
            for n, p in self.named_parameters():
                for fn in freeze_names:
                    if n.startswith(fn):
                        p.requires_grad = False
                        print(f"disable gradient of weights: {n}")
                        break

    def get_last_selfattention(self, source, level = 1, tgt_layer=None, padding_mask = None):
        features = self.feature_extractor(source)
        features = features.transpose(1, 2)
        features = self.layer_norm(features)
        if padding_mask is not None: # resize padding mask according to Conv pooling ratio
            extra = padding_mask.size(1) % features.size(1)
            if extra > 0:
                padding_mask = padding_mask[:, :-extra]
            padding_mask = padding_mask.view(padding_mask.size(0), features.size(1), -1)
            padding_mask = padding_mask.all(-1)
        if self.post_extract_proj is not None:
            features = self.post_extract_proj(features)
        x = self.dropout_input(features)

        x_conv = self.pos_conv(x.transpose(1, 2))
        x_conv = x_conv.transpose(1, 2)
        x += x_conv
        # ############################################
        if self.args.use_audio_cls_token:
            x = torch.cat([self.cls_token.repeat(x.shape[0],1,1), x], dim=1)
            if padding_mask != None:
                cls_token_padding_mask = torch.zeros((padding_mask.shape[0],1)).to(padding_mask)
                padding_mask = torch.cat([cls_token_padding_mask, padding_mask], dim=1)
        # ############################################
        if level == 1:
            _, attn_weights = self.encoder.extract_features(x, padding_mask=padding_mask, need_head_weights=True, tgt_layer=tgt_layer)
            return attn_weights
        else:
            assert self.args.trm2_layers != None
            features, layer_feats = self.encoder(x, padding_mask=padding_mask, tgt_layer=self.args.layer_use)
            # if self.args.double_cls:
            if self.args.use_audio_cls_token:
                cls_token = layer_feats[:,0]
                features = features[:,1:]
                layer_feats = layer_feats[:,1:]
                if padding_mask != None:
                    padding_mask = padding_mask[:,1:]
            else:
                cls_token = None
            x = layer_feats.transpose(1,2)
            x = self.feature_extractor2(x)
            x = x.transpose(1,2)
            x = self.layer_norm2(x)
            if padding_mask is not None: # resize padding mask according to Conv2 pooling ratio
                extra = padding_mask.size(1) % x.size(1)
                if extra > 0:
                    padding_mask = padding_mask[:, :-extra]
                padding_mask = padding_mask.view(padding_mask.size(0), x.size(1), -1)
                padding_mask = padding_mask.all(-1)
            
            x = self.post_extract_proj2(x)
            x = self.dropout_input2(x)

            if self.args.conv2_pos is not None:
                x_conv = self.pos_conv2(x.transpose(1, 2))
                x_conv = x_conv.transpose(1, 2)
                x += x_conv

            if self.args.use_audio_cls_token:
                cls_token = self.cls_token_proj(cls_token).unsqueeze(1)
                x = torch.cat([cls_token, x], dim=1)
                if padding_mask != None:
                    cls_token_padding_mask = torch.zeros((padding_mask.shape[0],1)).to(padding_mask)
                    padding_mask = torch.cat([cls_token_padding_mask, padding_mask], dim=1)

            _, attn_weights = self.encoder2.extract_features(x, padding_mask=padding_mask, tgt_layer=tgt_layer, need_head_weights=True)

            return attn_weights
            




class ConvFeatureExtractionModel(nn.Module):
    def __init__(
        self,
        conv_layers: List[Tuple[int, int, int]],
        dropout: float = 0.0,
        mode: str = "default",
        conv_bias: bool = False,
        first_conv = True
    ):
        super().__init__()

        assert mode in {"default", "layer_norm"}

        def block(
            n_in,
            n_out,
            k,
            stride,
            is_layer_norm=False,
            is_group_norm=False,
            conv_bias=False,
        ):
            def make_conv():
                conv = nn.Conv1d(n_in, n_out, k, stride=stride, bias=conv_bias)
                nn.init.kaiming_normal_(conv.weight)
                return conv

            assert (
                is_layer_norm and is_group_norm
            ) == False, "layer norm and group norm are exclusive"

            if is_layer_norm:
                return nn.Sequential(
                    make_conv(),
                    nn.Dropout(p=dropout),
                    nn.Sequential(
                        TransposeLast(),
                        Fp32LayerNorm(dim, elementwise_affine=True),
                        TransposeLast(),
                    ),
                    nn.GELU(),
                )
            elif is_group_norm:
                return nn.Sequential(
                    make_conv(),
                    nn.Dropout(p=dropout),
                    Fp32GroupNorm(dim, dim, affine=True),
                    nn.GELU(),
                )
            else:
                return nn.Sequential(make_conv(), nn.Dropout(p=dropout), nn.GELU())
        
        if first_conv:
            in_d = 1
        else:
            in_d = 768
        self.conv_layers = nn.ModuleList()
        for i, cl in enumerate(conv_layers):
            assert len(cl) == 3, "invalid conv definition: " + str(cl)
            (dim, k, stride) = cl

            self.conv_layers.append(
                block(
                    in_d,
                    dim,
                    k,
                    stride,
                    is_layer_norm=mode == "layer_norm",
                    is_group_norm=mode == "default" and i == 0,
                    conv_bias=conv_bias,
                )
            )
            in_d = dim

    def forward(self, x):

        # BxT -> BxCxT
        if len(x.shape) == 2:
            x = x.unsqueeze(1)

        for conv in self.conv_layers:
            x = conv(x)

        return x

# class ConvFeatureExtractionModel(nn.Module):
#     def __init__(
#         self,
#         conv_layers: List[Tuple[int, int, int]],
#         dropout: float = 0.0,
#         mode: str = "default",
#         conv_bias: bool = False,
#     ):
#         super().__init__()

#         assert mode in {"default", "layer_norm"}

#         def block(
#             n_in,
#             n_out,
#             k,
#             stride,
#             is_layer_norm=False,
#             is_group_norm=False,
#             conv_bias=False,
#         ):
#             def make_conv():
#                 conv = nn.Conv1d(n_in, n_out, k, stride=stride, bias=conv_bias)
#                 nn.init.kaiming_normal_(conv.weight)
#                 return conv

#             assert (
#                 is_layer_norm and is_group_norm
#             ) == False, "layer norm and group norm are exclusive"

#             if is_layer_norm:
#                 return nn.Sequential(
#                     make_conv(),
#                     nn.Dropout(p=dropout),
#                     nn.Sequential(
#                         TransposeLast(),
#                         Fp32LayerNorm(dim, elementwise_affine=True),
#                         TransposeLast(),
#                     ),
#                     nn.GELU(),
#                 )
#             elif is_group_norm:
#                 return nn.Sequential(
#                     make_conv(),
#                     nn.Dropout(p=dropout),
#                     Fp32GroupNorm(dim, dim, affine=True),
#                     nn.GELU(),
#                 )
#             else:
#                 return nn.Sequential(make_conv(), nn.Dropout(p=dropout), nn.GELU())

#         in_d = 1
#         self.conv_layers = nn.ModuleList()
#         for i, cl in enumerate(conv_layers):
#             assert len(cl) == 3, "invalid conv definition: " + str(cl)
#             (dim, k, stride) = cl

#             self.conv_layers.append(
#                 block(
#                     in_d,
#                     dim,
#                     k,
#                     stride,
#                     is_layer_norm=mode == "layer_norm",
#                     is_group_norm=mode == "default" and i == 0,
#                     conv_bias=conv_bias,
#                 )
#             )
#             in_d = dim

#     def forward(self, x):

#         # BxT -> BxCxT
#         x = x.unsqueeze(1)

#         for conv in self.conv_layers:
#             x = conv(x)

#         return x


class TransformerEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.layer_use = args.layer_use
        assert args.layer_use < args.encoder_layers, f"w2v2 only has {args.encoder_layers} layers, but you want layer feat from layer {args.layer_use+1}"
        self.dropout = args.dropout
        self.embedding_dim = args.encoder_embed_dim

        self.layers = nn.ModuleList(
            [
                TransformerSentenceEncoderLayer(
                    embedding_dim=self.embedding_dim,
                    ffn_embedding_dim=args.encoder_ffn_embed_dim,
                    num_attention_heads=args.encoder_attention_heads,
                    dropout=self.dropout,
                    attention_dropout=args.attention_dropout,
                    activation_dropout=args.activation_dropout,
                    activation_fn=args.activation_fn,
                    layer_norm_first=args.layer_norm_first,
                )
                for _ in range(args.encoder_layers)
            ]
        )

        self.layer_norm_first = args.layer_norm_first
        self.layer_norm = LayerNorm(self.embedding_dim)
        self.layerdrop = args.encoder_layerdrop

        self.apply(init_bert_params)

    def forward(self, x, padding_mask=None, superb=False, segment_layer=None, feature_layer=None, need_attention_weights=False, pre_feats=False):
        x = x.to(self.layer_norm.bias.data.dtype)
        if superb:
            assert not self.layer_norm_first
            all_feats = self.extract_features(x, padding_mask = padding_mask, all_hidden_states=True)
            return all_feats
        # print("1386", x.dtype)
        layer_feats, attn_weights = self.extract_features(x, padding_mask = padding_mask, segment_layer = segment_layer, feature_layer=feature_layer, need_head_weights=need_attention_weights)

        return layer_feats, attn_weights

    def extract_features(self, x, padding_mask=None, need_head_weights=False, segment_layer=None, feature_layer=None,  all_hidden_states=False, pre_feats=False):

        if not self.layer_norm_first:
            x = self.layer_norm(x)

        x = F.dropout(x, p=self.dropout, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)
        
        layer_feats = []
        end_layer = max(feature_layer + [segment_layer])
        for i, layer in enumerate(self.layers):
            x, z = layer(x, self_attn_padding_mask=padding_mask, need_weights=need_head_weights)
            if i in feature_layer:
                layer_feats.append(x.transpose(0, 1))
            if i == segment_layer:
                if len(z.shape) == 2:
                    z = z.unsqueeze(0)
                attn_weights = z # [bsz, num_heads, tgt_len, src_len]
            if i == end_layer:
                if len(z.shape) == 2: 
                    z = z.unsqueeze(0) 
                attn_weights = z # [bsz, num_heads, tgt_len, src_len]
                return layer_feats, attn_weights
        
        raise RuntimeError("shouldn't get here")
        if all_hidden_states:
            return layer_results
        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        return x, layer_feats

    def max_positions(self):
        """Maximum output length supported by the encoder."""
        return self.args.max_positions

    def upgrade_state_dict_named(self, state_dict, name):
        """Upgrade a (possibly old) state dict for new versions of fairseq."""
        return state_dict


class TransformerSentenceEncoderLayer(nn.Module):
    """
    Implements a Transformer Encoder Layer used in BERT/XLM style pre-trained
    models.
    """

    def __init__(
        self,
        embedding_dim: float = 768,
        ffn_embedding_dim: float = 3072,
        num_attention_heads: float = 8,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.1,
        activation_fn: str = "relu",
        layer_norm_first: bool = False,
    ) -> None:

        super().__init__()
        # Initialize parameters
        self.embedding_dim = embedding_dim
        self.dropout = dropout
        self.activation_dropout = activation_dropout

        # Initialize blocks
        self.activation_fn = utils.get_activation_fn(activation_fn)
        self.self_attn = MultiheadAttention(
            self.embedding_dim,
            num_attention_heads,
            dropout=attention_dropout,
            self_attention=True,
        )

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(self.activation_dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.layer_norm_first = layer_norm_first

        # layer norm associated with the self attention layer
        self.self_attn_layer_norm = LayerNorm(self.embedding_dim)
        self.fc1 = nn.Linear(self.embedding_dim, ffn_embedding_dim)
        self.fc2 = nn.Linear(ffn_embedding_dim, self.embedding_dim)

        # layer norm associated with the position wise feed-forward NN
        self.final_layer_norm = LayerNorm(self.embedding_dim)

    def forward(
        self,
        x: torch.Tensor,
        self_attn_mask: torch.Tensor = None,
        self_attn_padding_mask: torch.Tensor = None,
        need_weights: bool = False,
        need_head_weights: bool = False,
        att_args=None,
    ):
        """
        LayerNorm is applied either before or after the self-attention/ffn
        modules similar to the original Transformer imlementation.
        """
        residual = x

        if self.layer_norm_first:
            x = self.self_attn_layer_norm(x)
            temp = self.self_attn(
                query=x,
                key=x,
                value=x,
                key_padding_mask=self_attn_padding_mask,
                need_weights=need_weights,
                attn_mask=self_attn_mask,
            )
            if len(temp) == 3:
                x, attn, _  = temp
            elif len(temp) == 2:
                x, attn = temp
            else:
                print(f"length of self_attn should be either 2 or 3, but it's {len(temp)}")
            x = self.dropout1(x)
            x = residual + x

            residual = x
            x = self.final_layer_norm(x)
            x = self.activation_fn(self.fc1(x))
            x = self.dropout2(x)
            x = self.fc2(x)
            x = self.dropout3(x)
            x = residual + x
        else:
            temp = self.self_attn(
                query=x,
                key=x,
                value=x,
                key_padding_mask=self_attn_padding_mask,
                need_weights=need_weights
            )
            if len(temp) == 3:
                x, attn, _  = temp
            elif len(temp) == 2:
                x, attn = temp
            else:
                print(f"length of self_attn should be either 2 or 3, but it's {len(temp)}")

            x = self.dropout1(x)
            x = residual + x

            x = self.self_attn_layer_norm(x)

            residual = x
            x = self.activation_fn(self.fc1(x))
            x = self.dropout2(x)
            x = self.fc2(x)
            x = self.dropout3(x)
            x = residual + x
            x = self.final_layer_norm(x)

        return x, attn


