import logging
import math
from collections import OrderedDict
from functools import partial
from typing import Dict, List, Mapping, NamedTuple, Optional, Union, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .ops.ats import AdaptiveTokenSampler

NORMALIZE_L2 = "l2"

LayerNorm = partial(nn.LayerNorm, eps=1e-6)

from .vision_transformer import (MLPBlock,
                                 Encoder,
                                 EncoderBlock,
                                 VisionTransformer,
                                 is_pos_int,
                                 ConvStemLayer,
                                 get_same_padding_for_kernel_size,
                                 VisionTransformerHead)


class ATSBlock(nn.Module):
    """
    Transformer Block + ATS
    """

    def __init__(
        self,
        num_heads, 
        hidden_dim,
        mlp_dim, 
        dropout_rate, 
        attention_dropout_rate,
        drop_tokens=False,
    ):
        super().__init__()
        
        self.ln_1 = LayerNorm(hidden_dim)
        self.self_attention = AdaptiveTokenSampler(
            hidden_dim,
            num_heads,
            attention_dropout_rate,
            drop_tokens=drop_tokens,
        )
        self.dropout = nn.Dropout(dropout_rate)
        self.ln_2 = LayerNorm(hidden_dim)
        self.mlp = MLPBlock(hidden_dim, mlp_dim, dropout_rate)
        self.num_heads = num_heads

    def forward(
        self,
        input,
        n_tokens,
        policy: Tensor = None,
        sampler: Tensor = None,
        n_ref_tokens: int = 197,
    ):
        x = self.ln_1(input)
        x_out, _, selected_x, policy, sampler = self.self_attention(
            query=x, 
            key=x, 
            value=x, 
            need_weights=False,
            policy=policy,
            sampler=sampler,
            n_tokens=n_tokens,
            raw_x=x.transpose(0, 1),
            n_ref_tokens=n_ref_tokens,
        )
        x = selected_x.transpose(0, 1) + self.dropout(x_out)
        x = x * policy.transpose(0, 1)
        out = self.mlp(self.ln_2(x))
        x = x + self.dropout(out)
        x = x * policy.transpose(0, 1)
        return x, policy


class ATSEncoder(nn.Module):
    """ATS Transformer Encoder."""

    def __init__(
        self,
        seq_length,
        num_layers,
        num_heads,
        hidden_dim,
        mlp_dim,
        dropout_rate,
        attention_dropout_rate,
        ats_blocks=[],
        num_tokens=197,    # num of tokens to be sampled
        drop_tokens=False,
    ):
        super().__init__()
        self.pos_embedding = nn.Parameter(
            torch.empty(seq_length, 1, hidden_dim).normal_(std=0.02)
        )
        self.dropout = nn.Dropout(dropout_rate)
        self.ats_blocks = ats_blocks
        self.num_tokens = num_tokens
        layers = []
        for i in range(num_layers):
            if i in ats_blocks:
                layers.append(
                    (
                        f"layer_{i}",
                        ATSBlock(
                            num_heads,
                            hidden_dim,
                            mlp_dim,
                            dropout_rate,
                            attention_dropout_rate,
                            drop_tokens=drop_tokens,
                        ),
                    )
                )
            else:
                layers.append(
                    (
                        f"layer_{i}",
                        EncoderBlock(
                            num_heads,
                            hidden_dim,
                            mlp_dim,
                            dropout_rate,
                            attention_dropout_rate,
                        ),
                    )
                )
        self.layers = nn.Sequential(OrderedDict(layers))
        self.ln = LayerNorm(hidden_dim)

    def forward(self, x):
        x = x + self.pos_embedding  # should broadcast to the same shape
        x = self.dropout(x)

        B = x.shape[1]
        init_n = x.shape[0]
        policies = []
        policy = torch.ones(B, init_n, 1, dtype=x.dtype, device=x.device)
        sampler = torch.nonzero(policy)
        for i, layer in enumerate(self.layers):
            if i in self.ats_blocks:
                x, policy = layer(
                    x,
                    n_tokens=self.num_tokens,
                    policy=policy,
                    sampler=sampler,
                    n_ref_tokens=init_n,
                )
            else:
                x = layer(x)

        return self.ln(x)


class ATSVisionTransformer(VisionTransformer):
    def __init__(
        self, 
        image_size, 
        patch_size, 
        num_layers, 
        num_heads, 
        hidden_dim, 
        mlp_dim, 
        dropout_rate=0, 
        attention_dropout_rate=0, 
        classifier="token", 
        conv_stem_layers: Union[List[ConvStemLayer], List[Dict], None] = None, 
        num_classes: Optional[int] = None,
        ats_blocks=[],
        num_tokens=197,    # num of tokens to be sampled
        drop_tokens=False,):
        super().__init__(
            image_size, 
            patch_size, 
            num_layers, 
            num_heads, 
            hidden_dim, 
            mlp_dim, 
            dropout_rate, 
            attention_dropout_rate, 
            classifier, 
            conv_stem_layers, 
            num_classes)

        seq_length = (image_size // patch_size) ** 2
        if self.classifier == "token":
            # add a class token
            self.class_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
            seq_length += 1

        self.encoder = ATSEncoder(
            seq_length,
            num_layers,
            num_heads,
            hidden_dim,
            mlp_dim,
            dropout_rate,
            attention_dropout_rate,
            ats_blocks=ats_blocks,
            num_tokens=num_tokens,
            drop_tokens=drop_tokens,
        )
        self.init_weights()

class ATSViTB16(ATSVisionTransformer):
    def __init__(
        self,
        image_size=224,
        dropout_rate=0,
        attention_dropout_rate=0,
        classifier="token",
        num_classes=None,
        ats_blocks=[],
        num_tokens=197,    # num of tokens to be sampled
        drop_tokens=False,
        **kwargs,
    ):
        super().__init__(
            image_size=image_size,
            patch_size=16,
            num_layers=12,
            num_heads=12,
            hidden_dim=768,
            mlp_dim=3072,
            dropout_rate=dropout_rate,
            attention_dropout_rate=attention_dropout_rate,
            classifier=classifier,
            num_classes=num_classes,
            ats_blocks=ats_blocks,
            num_tokens=num_tokens,
            drop_tokens=drop_tokens,
        )

class ATSViTL16(ATSVisionTransformer):
    def __init__(
        self,
        image_size=224,
        dropout_rate=0,
        attention_dropout_rate=0,
        classifier="token",
        num_classes=None,
        ats_blocks=[],
        num_tokens=197,    # num of tokens to be sampled
        drop_tokens=False,
        **kwargs,
    ):
        super().__init__(
            image_size=image_size,
            patch_size=16,
            num_layers=24,
            num_heads=16,
            hidden_dim=1024,
            mlp_dim=4096,
            dropout_rate=dropout_rate,
            attention_dropout_rate=attention_dropout_rate,
            classifier=classifier,
            num_classes=num_classes,
            ats_blocks=ats_blocks,
            num_tokens=num_tokens,
            drop_tokens=drop_tokens,
        )

class ATSViTH14(ATSVisionTransformer):
    def __init__(
        self,
        image_size=224,
        dropout_rate=0,
        attention_dropout_rate=0,
        classifier="token",
        num_classes=None,
        ats_blocks=[],
        num_tokens=197,    # num of tokens to be sampled
        drop_tokens=False,
        **kwargs,
    ):
        super().__init__(
            image_size=image_size,
            patch_size=14,
            num_layers=32,
            num_heads=16,
            hidden_dim=1280,
            mlp_dim=5120,
            dropout_rate=dropout_rate,
            attention_dropout_rate=attention_dropout_rate,
            classifier=classifier,
            num_classes=num_classes,
            ats_blocks=ats_blocks,
            num_tokens=num_tokens,
            drop_tokens=drop_tokens,
        )
