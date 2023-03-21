# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Modified from https://github.com/facebookresearch/ClassyVision/blob/main/classy_vision/models/vision_transformer.py

"""
Vision Transformer implementation from https://arxiv.org/abs/2010.11929.
References:
https://github.com/google-research/vision_transformer
https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
"""

import math
from collections import OrderedDict
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .vision_transformer import VisionTransformer, EncoderBlock

def complement_idx(idx, dim):
    """
    Compute the complement: set(range(dim)) - set(idx).
    idx is a multi-dimensional tensor, find the complement for its trailing dimension,
    all other dimension is considered batched.
    Args:
        idx: input index, shape: [N, *, K]
        dim: the max index for complement
    """
    a_tensor = torch.arange(dim, device=idx.device)
    ndim = idx.ndim
    dims = idx.shape
    n_idx = dims[-1]
    dims = dims[:-1] + (-1,)
    for i in range(1, ndim):
        a_tensor = a_tensor.unsqueeze(0)
    # a_tensor.shape:[128, 576]
    a_tensor = a_tensor.expand(*dims)

    masked = torch.scatter(a_tensor, -1, idx, 0)
    compl, _ = torch.sort(masked, dim=-1, descending=False)
    compl = compl.permute(-1, *tuple(range(ndim - 1)))
    compl = compl[n_idx:].permute(*(tuple(range(1, ndim)) + (0,)))
    return compl

class MultiheadAttention(nn.MultiheadAttention):
    def __init__(self, *args, keep_rate=1.0, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
        self.keep_rate = keep_rate
        assert 0 < keep_rate <= 1, "keep_rate must > 0 and <= 1, got {0}".format(keep_rate)

    def forward(
        self,
        query,
        key,
        value,
        key_padding_mask=None,
        need_weights=True,
        attn_mask=None,
        keep_rate=None,
        tokens=None,
    ):

        if keep_rate is None:
            keep_rate = self.keep_rate

        if self.batch_first:
            query, key, value = [x.transpose(1, 0) for x in (query, key, value)]
            B, N, C = query.shape
        else:
            N, B, C = query.shape
        if not self._qkv_same_embed_dim:
            attn_output, attn_output_weights = F.multi_head_attention_forward(
                query,
                key,
                value,
                self.embed_dim,
                self.num_heads,
                self.in_proj_weight,
                self.in_proj_bias,
                self.bias_k,
                self.bias_v,
                self.add_zero_attn,
                self.dropout,
                self.out_proj.weight,
                self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask,
                need_weights=True,
                attn_mask=attn_mask,
                use_separate_proj_weight=True,
                q_proj_weight=self.q_proj_weight,
                k_proj_weight=self.k_proj_weight,
                v_proj_weight=self.v_proj_weight,
            )
        else:
            attn_output, attn_output_weights = F.multi_head_attention_forward(
                query,
                key,
                value,
                self.embed_dim,
                self.num_heads,
                self.in_proj_weight,
                self.in_proj_bias,
                self.bias_k,
                self.bias_v,
                self.add_zero_attn,
                self.dropout,
                self.out_proj.weight,
                self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask,
                need_weights=True,
                attn_mask=attn_mask,
            )
        if self.batch_first:
            x = attn_output.transpose(1, 0)
        else:
            x = attn_output

        left_tokens = N - 1
        # new_evit,c
        if (
            self.keep_rate < 1 and keep_rate < 1 or tokens is not None
        ):  # double check the keep rate
            left_tokens = math.ceil(keep_rate * (N - 1))
            if tokens is not None:
                left_tokens = tokens
            if left_tokens == N - 1:
                return x, None, None, None, left_tokens
            assert left_tokens >= 1

            cls_attn = attn_output_weights[:, :, 1:]  # [B, H, N-1]
            cls_attn = cls_attn.mean(dim=1)  # [B, N-1]
            _, idx = torch.topk(
                cls_attn, left_tokens, dim=1, largest=True, sorted=True
            )  # [B, left_tokens]
            # cls_idx = torch.zeros(B, 1, dtype=idx.dtype, device=idx.device)
            # index = torch.cat([cls_idx, idx + 1], dim=1)
            index = idx.unsqueeze(-1).expand(-1, -1, C)  # [B, left_tokens, C]

            return x, index, idx, cls_attn, left_tokens
        else:
            return x, None, None, None, left_tokens


class EViTEncoderBlock(EncoderBlock):
    """Transformer encoder block.
    From @myleott -
    There are at least three common structures.
    1) Attention is all you need had the worst one, where the layernorm came after each
        block and was in the residual path.
    2) BERT improved upon this by moving the layernorm to the beginning of each block
        (and adding an extra layernorm at the end).
    3) There's a further improved version that also moves the layernorm outside of the
        residual path, which is what this implementation does.
    Figure 1 of this paper compares versions 1 and 3:
        https://openreview.net/pdf?id=B1x8anVFPr
    Figure 7 of this paper compares versions 2 and 3 for BERT:
        https://arxiv.org/abs/1909.08053
    """

    def __init__(
        self, 
        num_heads, 
        hidden_dim, 
        mlp_dim, 
        dropout_rate, 
        attention_dropout_rate,
        keep_rate=0.0,
        fuse_token=False,
    ):
        super().__init__(
                num_heads, 
                hidden_dim, 
                mlp_dim, 
                dropout_rate, 
                attention_dropout_rate)
        
        self.self_attention = MultiheadAttention(
            hidden_dim, num_heads, dropout=attention_dropout_rate, keep_rate=keep_rate
        )  # uses correct initialization by default
        self.keep_rate = keep_rate
        self.fuse_token = fuse_token

    def forward(self, x, keep_rate=None, tokens=None):
        if keep_rate is None:
            keep_rate = (
                self.keep_rate
            )  # this is for inference, use the default keep rate

        ori_input = x
        N, B, C = x.shape
        x = self.ln_1(x)
        tmp, index, idx, cls_attn, left_tokens = self.self_attention(
            query=x, key=x, value=x, need_weights=True, keep_rate=keep_rate,
        )
        x = ori_input + self.dropout(tmp)

        if index is not None:
            x = x.transpose(0, 1)
            # B, N, C = x.shape
            non_cls = x[:, 1:]
            x_others = torch.gather(non_cls, dim=1, index=index)  # [B, left_tokens, C]

            if self.fuse_token:
                compl = complement_idx(idx, N - 1)  # [B, N-1-left_tokens]
                non_topk = torch.gather(
                    non_cls, dim=1, index=compl.unsqueeze(-1).expand(-1, -1, C)
                )  # [B, N-1-left_tokens, C]

                non_topk_attn = torch.gather(
                    cls_attn, dim=1, index=compl
                )  # [B, N-1-left_tokens]
                extra_token = torch.sum(
                    non_topk * non_topk_attn.unsqueeze(-1), dim=1, keepdim=True
                )  # [B, 1, C]
                x = torch.cat([x[:, 0:1], x_others, extra_token], dim=1)
            else:
                x = torch.cat([x[:, 0:1], x_others], dim=1)
            x = x.transpose(0, 1)

        return x + self.mlp(self.ln_2(x))


class EViT(VisionTransformer):
    def __init__(self, 
        image_size, 
        patch_size, 
        num_layers, 
        num_heads, 
        hidden_dim, 
        mlp_dim, 
        dropout_rate=0, 
        attention_dropout_rate=0, 
        classifier="token", 
        conv_stem_layers=None, 
        num_classes: Optional[int] = None, 
        keep_rate=(1, ), 
        fuse_token=False
    ):
        super().__init__(
                image_size, 
                patch_size, 
                num_layers, 
                num_heads, 
                hidden_dim, 
                mlp_dim, 
                dropout_rate, attention_dropout_rate, classifier, 
                conv_stem_layers, 
                num_classes)

        layers = []
        for i in range(num_layers):
            layers.append(
                (
                    f"layer_{i}",
                    EViTEncoderBlock(
                        num_heads,
                        hidden_dim,
                        mlp_dim,
                        dropout_rate,
                        attention_dropout_rate,
                        keep_rate=keep_rate[i],
                        fuse_token=fuse_token,
                    ),
                )
            )
        self.encoder.layers = nn.Sequential(OrderedDict(layers))


class EViTB16(EViT):
    def __init__(
        self,
        image_size=224,
        dropout_rate=0,
        attention_dropout_rate=0,
        classifier="token",
        num_classes=None,
        base_keep_rate = 0.6,
        drop_loc = (3, 6, 9),
        fuse_token=True,
        **kwargs
    ):
        keep_rate = [1] * 12      
        for loc in drop_loc:
            keep_rate[loc] = base_keep_rate

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
            keep_rate=keep_rate,
            fuse_token=fuse_token,
        )


class EViTL16(EViT):
    def __init__(
        self,
        image_size=224,
        dropout_rate=0,
        attention_dropout_rate=0,
        classifier="token",
        num_classes=None,
        base_keep_rate = 0.6,
        drop_loc = (6, 12, 18),
        fuse_token=True,
        **kwargs,
    ):
        
        keep_rate = [1] * 24
        for loc in drop_loc:
            keep_rate[loc] = base_keep_rate
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
            keep_rate=keep_rate,
            fuse_token=fuse_token,
        )


class EViTH14(EViT):
    def __init__(
        self,
        image_size=224,
        dropout_rate=0,
        attention_dropout_rate=0,
        classifier="token",
        num_classes=None,
        base_keep_rate = 0.6,
        drop_loc = (8, 16, 24),
        fuse_token=True,
        **kwargs
    ):

        keep_rate = [1] * 32
        for loc in drop_loc:
            keep_rate[loc] = base_keep_rate
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
            keep_rate=keep_rate,
            fuse_token=fuse_token,
        )
