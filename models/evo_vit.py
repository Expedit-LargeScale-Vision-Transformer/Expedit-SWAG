import math

import torch

from .vision_transformer import VisionTransformer

import math
from collections import OrderedDict
from functools import partial

import torch
import torch.nn as nn

from .vision_transformer import VisionTransformer, EncoderBlock

LayerNorm = partial(nn.LayerNorm, eps=1e-6)

def easy_gather(x, indices):
    # x: B,N,C; indices: B,N
    B, N, C = x.shape
    N_new = indices.shape[1]
    offset = torch.arange(B, dtype=torch.long, device=x.device).view(B, 1) * N
    indices = indices + offset
    out = x.reshape(B * N, C)[indices.view(-1)].reshape(B, N_new, C)
    return out

class EvoEncoderBlock(EncoderBlock):
    def __init__(self, num_heads, hidden_dim, mlp_dim, dropout_rate, attention_dropout_rate, prune_ratio, tradeoff):
        super().__init__(num_heads, hidden_dim, mlp_dim, dropout_rate, attention_dropout_rate)

        self.prune_ratio = prune_ratio
        self.tradeoff = tradeoff

    def forward(self, x, cls_attn=None):
        if self.prune_ratio != 1:
            x = x.transpose(0, 1)
            x_patch = x[:, 1:, :]

            B, N, C = x_patch.shape
            N_ = int(N * self.prune_ratio)
            indices = torch.argsort(cls_attn, dim=1, descending=True)
            x_patch = torch.cat((x_patch, cls_attn.unsqueeze(-1)), dim=-1)
            x_sorted = easy_gather(x_patch, indices)
            x_patch, cls_attn = x_sorted[:, :, :-1], x_sorted[:, :, -1]

            if self.training:
                x_ = torch.cat((x[:, :1, :], x_patch), dim=1)
            else:
                x[:, 1:, :] = x_patch
                x_ = x
            x = x_[:, :N_ + 1]
            x_ = x_.transpose(0, 1)
            x = x.transpose(0, 1)

            # slow updating
            tmp_x = x
            B, N, C = x.shape
            # x = self.norms[index](x)
            # v = self.vs[index](x)
            # attn = self.qks[index](x)
            x = self.ln_1(x)
            x, attn = self.self_attention(query=x, key=x, value=x, need_weights=True)
            x = self.dropout(x)
            x = x + tmp_x

            # with torch.no_grad():
            if self.training:
                temp_cls_attn = (1 - self.tradeoff) * cls_attn[:, :N_] + self.tradeoff * attn[:, 0, 1:]
                cls_attn = torch.cat((temp_cls_attn, cls_attn[:, N_:]), dim=1)

            else:
                cls_attn[:, :N_] = (1 - self.tradeoff) * cls_attn[:, :N_] + self.tradeoff * attn[:, 0, 1:]

            # x = (attn @ v).transpose(1, 2).reshape(B, N, C)
            # x = self.projs[index](x)
            # x = blk.drop_path(x)
            # x = x + tmp_x

            y = self.ln_2(x)
            y = self.mlp(y)
            x = x + y

            # fast updating, only preserving the placeholder tokens presents enough good results on DeiT
            if self.training:
                x = torch.cat((x, x_[N_ + 1:]), dim=1)
            else:
                x_[:N_ + 1] = x
                x = x_
        else:
            tmp_x = x
            x = self.ln_1(x)
            x, attn = self.self_attention(query=x, key=x, value=x, need_weights=True)
            x = self.dropout(x)
            x = x + tmp_x

            if cls_attn == None:
                cls_attn = attn[:, 0, 1:]
            else:
                cls_attn = (1 - self.tradeoff) * cls_attn + self.tradeoff * attn[:, 0, 1:]
            
            y = self.ln_2(x)
            y = self.mlp(y)
            x = x + y

        return x, cls_attn


class EvoEncoder(nn.Module):
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
        prune_ratio,
        tradeoff,
        prune_location,
    ):
        super().__init__()
        self.pos_embedding = nn.Parameter(
            torch.empty(seq_length, 1, hidden_dim).normal_(std=0.02)
        )
        self.dropout = nn.Dropout(dropout_rate)

        if not isinstance(prune_ratio, (list, tuple)):
            prune_ratio = [1.0] * prune_location + [prune_ratio] * (num_layers - prune_location)
        if not isinstance(tradeoff, (list, tuple)):
            tradeoff = [tradeoff] * num_layers

        layers = []
        for i in range(num_layers):
            layers.append(
                (
                    f"layer_{i}",
                    EvoEncoderBlock(
                        num_heads,
                        hidden_dim,
                        mlp_dim,
                        dropout_rate,
                        attention_dropout_rate,
                        prune_ratio[i],
                        tradeoff[i],
                    ),
                )
            )
        self.layers = nn.Sequential(OrderedDict(layers))
        self.ln = LayerNorm(hidden_dim)

    def forward(self, x):
        x = x + self.pos_embedding  # should broadcast to the same shape
        x = self.dropout(x)
        cls_attn = None
        for layer in self.layers:
            x, cls_attn = layer(x, cls_attn)
        return self.ln(x)


class EvoViT(VisionTransformer):
    """Vision Transformer as per https://arxiv.org/abs/2010.11929."""

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
        prune_ratio=1.,
        tradeoff=0.5,
        prune_location=3,
        **kwargs,
    ):
        super().__init__(
            image_size,
            patch_size,
            num_layers,
            num_heads,
            hidden_dim,
            mlp_dim,
            dropout_rate=0,
            attention_dropout_rate=0,
            **kwargs)

        seq_length = (image_size // patch_size) ** 2
        if self.classifier == "token":
            # add a class token
            self.class_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
            seq_length += 1
            
        self.encoder = EvoEncoder(
            seq_length,
            num_layers,
            num_heads,
            hidden_dim,
            mlp_dim,
            dropout_rate,
            attention_dropout_rate,
            prune_ratio=prune_ratio,
            tradeoff=tradeoff,
            prune_location=prune_location,
        )
        
class EvoViTB16(EvoViT):
    def __init__(
        self,
        image_size=224,
        dropout_rate=0,
        attention_dropout_rate=0,
        classifier="token",
        num_classes=None,
        prune_ratio=1.,
        tradeoff=0.5,
        prune_location=3,
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
            prune_ratio=prune_ratio,
            tradeoff=tradeoff,
            prune_location=prune_location,
        )

class EvoViTL16(EvoViT):
    def __init__(
        self,
        image_size=224,
        dropout_rate=0,
        attention_dropout_rate=0,
        classifier="token",
        num_classes=None,
        prune_ratio=1.,
        tradeoff=0.5,
        prune_location=3,
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
            prune_ratio=prune_ratio,
            tradeoff=tradeoff,
            prune_location=prune_location,
        )

class EvoViTH14(EvoViT):
    def __init__(
        self,
        image_size=224,
        dropout_rate=0,
        attention_dropout_rate=0,
        classifier="token",
        num_classes=None,
        prune_ratio=1.,
        tradeoff=0.5,
        prune_location=3,
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
            prune_ratio=prune_ratio,
            tradeoff=tradeoff,
            prune_location=prune_location,
        )

