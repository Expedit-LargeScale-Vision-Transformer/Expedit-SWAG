import logging
import math
from collections import OrderedDict
from functools import partial
from typing import Dict, List, Mapping, NamedTuple, Optional, Union

import torch
import torch.nn as nn

from ACT import AdaMultiheadAttention

NORMALIZE_L2 = "l2"

LayerNorm = partial(nn.LayerNorm, eps=1e-6)

from .vision_transformer import (MLPBlock,
                                 EncoderBlock,
                                 VisionTransformer,
                                 is_pos_int,
                                 ConvStemLayer,
                                 get_same_padding_for_kernel_size,
                                 VisionTransformerHead)


class ACTEncoderBlock(nn.Module):
    def __init__(self,
                 num_heads,
                 hidden_dim,
                 mlp_dim,
                 dropout_rate,
                 attention_dropout_rate,
                 group_Q=True,
                 group_K=False,
                 q_hashes=32,
                 k_hashes=32
                 ):
        super().__init__()
        self.ln_1 = LayerNorm(hidden_dim)

        self.self_attention = AdaMultiheadAttention(embed_dim=hidden_dim,
                                                    num_heads=num_heads,
                                                    group_Q=group_Q,
                                                    group_K=group_K,
                                                    q_hashes=q_hashes,
                                                    k_hashes=k_hashes,
                                                    dropout=attention_dropout_rate)  # uses correct initialization by default
        self.dropout = nn.Dropout(dropout_rate)
        self.ln_2 = LayerNorm(hidden_dim)
        self.mlp = MLPBlock(hidden_dim, mlp_dim, dropout_rate)
        self.num_heads = num_heads

    def forward(self, input):
        x = self.ln_1(input)
        x, _ = self.self_attention(query=x, key=x, value=x)
        x = self.dropout(x)
        x = x + input

        y = self.ln_2(x)
        y = self.mlp(y)
        return x + y

    def flops(self, x):
        raise NotImplementedError

    def activations(self, out, x):
        raise NotImplementedError


class WrappedEncoderBlock(EncoderBlock):
    def __init__(self,
                 num_heads,
                 hidden_dim,
                 mlp_dim,
                 dropout_rate,
                 attention_dropout_rate,
                 **kwargs):
        super(WrappedEncoderBlock, self).__init__(num_heads,
                                                  hidden_dim,
                                                  mlp_dim,
                                                  dropout_rate,
                                                  attention_dropout_rate)


class ActEncoder(nn.Module):
    """Transformer Encoder."""

    def __init__(
            self,
            seq_length,
            num_layers,
            num_heads,
            hidden_dim,
            mlp_dim,
            dropout_rate,
            attention_dropout_rate,
            act_plug_in_index=0,
            act_plug_out_index=-1,
            group_Q=True,
            group_K=False,
            q_hashes=32,
            k_hashes=32
    ):
        super().__init__()
        self.pos_embedding = nn.Parameter(
            torch.empty(seq_length, 1, hidden_dim).normal_(std=0.02)
        )
        self.dropout = nn.Dropout(dropout_rate)
        layers = []
        block = WrappedEncoderBlock
        for i in range(num_layers):
            if i == act_plug_in_index:
                block = ACTEncoderBlock
            elif i == act_plug_out_index:
                block = WrappedEncoderBlock
            layers.append(
                (
                    f"layer_{i}",
                    block(
                        num_heads,
                        hidden_dim,
                        mlp_dim,
                        dropout_rate,
                        attention_dropout_rate,
                        group_Q=group_Q,
                        group_K=group_K,
                        q_hashes=q_hashes,
                        k_hashes=k_hashes
                    ),
                )
            )
        self.layers = nn.Sequential(OrderedDict(layers))
        self.ln = LayerNorm(hidden_dim)

    def forward(self, x):
        x = x + self.pos_embedding  # should broadcast to the same shape
        return self.ln(self.layers(self.dropout(x)))


class ActVisionTransformer(VisionTransformer):
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
            act_plug_in_index=0,
            act_plug_out_index=-1,
            act_group_q=True,
            act_group_k=False,
            act_q_hashes=32,
            act_k_hashes=32,
            classifier="token",
            conv_stem_layers=None,
            num_classes: Optional[int] = None,
            **kwargs,
    ):
        super(VisionTransformer, self).__init__()
        assert image_size % patch_size == 0, "Input shape indivisible by patch size"
        assert classifier in ["token", "gap"], "Unexpected classifier mode"
        assert num_classes is None or is_pos_int(num_classes)

        self.image_size = image_size
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
        self.mlp_dim = mlp_dim
        self.attention_dropout_rate = attention_dropout_rate
        self.dropout_rate = dropout_rate
        self.classifier = classifier

        input_channels = 3

        self.conv_stem_layers = conv_stem_layers
        if conv_stem_layers is None:
            # conv_proj is a more efficient version of reshaping, permuting and projecting
            # the input
            self.conv_proj = nn.Conv2d(
                input_channels, hidden_dim, kernel_size=patch_size, stride=patch_size
            )
        else:
            prev_channels = input_channels
            self.conv_proj = nn.Sequential()
            for i, conv_stem_layer in enumerate(conv_stem_layers):
                if isinstance(conv_stem_layer, Mapping):
                    conv_stem_layer = ConvStemLayer(**conv_stem_layer)
                kernel = conv_stem_layer.kernel
                stride = conv_stem_layer.stride
                out_channels = conv_stem_layer.out_channels
                padding = get_same_padding_for_kernel_size(kernel)
                self.conv_proj.add_module(
                    f"conv_{i}",
                    nn.Conv2d(
                        prev_channels,
                        out_channels,
                        kernel_size=kernel,
                        stride=stride,
                        padding=padding,
                        bias=False,
                    ),
                )
                self.conv_proj.add_module(f"bn_{i}", nn.BatchNorm2d(out_channels))
                self.conv_proj.add_module(f"relu_{i}", nn.ReLU())
                prev_channels = out_channels
            self.conv_proj.add_module(
                f"conv_{i + 1}", nn.Conv2d(prev_channels, hidden_dim, kernel_size=1)
            )

        seq_length = (image_size // patch_size) ** 2
        if self.classifier == "token":
            # add a class token
            self.class_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
            seq_length += 1

        self.encoder = ActEncoder(
            seq_length,
            num_layers,
            num_heads,
            hidden_dim,
            mlp_dim,
            dropout_rate,
            attention_dropout_rate,
            act_plug_in_index,
            act_plug_out_index,
            act_group_q,
            act_group_k,
            act_q_hashes,
            act_k_hashes,
        )
        self.trunk_output = nn.Identity()

        self.seq_length = seq_length
        self.init_weights()

        if num_classes is not None:
            self.head = VisionTransformerHead(
                num_classes=num_classes, in_plane=hidden_dim
            )
        else:
            self.head = None


class ActViTL16(ActVisionTransformer):
    def __init__(
            self,
            image_size=224,
            dropout_rate=0,
            attention_dropout_rate=0,
            classifier="token",
            num_classes=None,
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
            **kwargs
        )


class ActViTH14(ActVisionTransformer):
    def __init__(
            self,
            image_size=224,
            dropout_rate=0,
            attention_dropout_rate=0,
            classifier="token",
            num_classes=None,
            **kwargs
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
            **kwargs
        )
