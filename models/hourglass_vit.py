import math

import torch

from .vision_transformer import VisionTransformer
from .ops.cluster import TokenClusteringBlock

def reshape_as_aspect_ratio(x, ratio, channel_last=False):
    assert x.ndim == 3
    B, N, C = x.size()
    s = round(math.sqrt(N / (ratio[0] * ratio[1])))
    perm = (0, 1, 2) if channel_last else (0, 2, 1)
    return x.permute(*perm).view(B, C, s * ratio[0], s * ratio[1])

def get_aspect_ratio(x, y):
    gcd = math.gcd(x, y)
    return x // gcd, y // gcd

class HourglassViT(VisionTransformer):
    """Vision Transformer as per https://arxiv.org/abs/2010.11929."""

    def __init__(
        self,
        cluster_after_output=True,
        clustering_location=0,
        num_cluster=256,
        cluster_iters=5,
        cluster_temperture=1.,
        cluster_window_size=5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.token_clustering_layer = TokenClusteringBlock(
            num_cluster,
            cluster_iters,
            cluster_temperture,
            cluster_window_size,
        )
        self.cluster_after_output = cluster_after_output
        self.clustering_location = clustering_location

        # self.token_reconstruction_layer = TSAUnpooling(k=unpool_k, temperture=0.2)
        
    def cluster(self, x, aspect_ratio):
        x = x.permute(1, 0, 2)  # B, L, C
        # unpooler.update_state(feat_before_pooling=x[:, 1:])
        cls_tokens = x[:, 0:1]
        x = reshape_as_aspect_ratio(x[:, 1:], aspect_ratio)
        x, hard_labels = self.token_clustering_layer(x)
        x = torch.cat([cls_tokens, x], dim=1)
        # unpooler.update_state(feat_after_pooling=x[:, 1:])
        x = x.permute(1, 0, 2)  # L, B, C
        return x

    def forward(self, x: torch.Tensor):
        assert x.ndim == 4, "Unexpected input shape"
        n, c, h, w = x.shape
        p = self.patch_size
        assert h == w == self.image_size
        n_h = h // p
        n_w = w // p

        # (n, c, h, w) -> (n, hidden_dim, n_h, n_w)
        x = self.conv_proj(x)
        # (n, hidden_dim, n_h, n_w) -> (n, hidden_dim, (n_h * n_w))
        x = x.reshape(n, self.hidden_dim, n_h * n_w)

        # (n, hidden_dim, (n_h * n_w)) -> ((n_h * n_w), n, hidden_dim)
        # the self attention layer expects inputs in the format (S, N, E)
        # where S is the source sequence length, N is the batch size, E is the
        # embedding dimension
        x = x.permute(2, 0, 1)

        if self.classifier == "token":
            # expand the class token to the full batch
            batch_class_token = self.class_token.expand(-1, n, -1)
            x = torch.cat([batch_class_token, x], dim=0)

        x = x + self.encoder.pos_embedding
        for i, layer in enumerate(self.encoder.layers):
            if not self.cluster_after_output and i == self.clustering_location:
                x = self.cluster(x, get_aspect_ratio(n_h, n_w))
            x = layer(self.encoder.dropout(x))
            if self.cluster_after_output and i == self.clustering_location:
                x = self.cluster(x, get_aspect_ratio(n_h, n_w))
        x = self.encoder.ln(x)

        if self.classifier == "token":
            # just return the output for the class token
            x = x[0, :, :]
        else:
            x = x.mean(dim=0)

        x = self.trunk_output(x)
        if self.head is not None:
            x = self.head(x)

        return x

class HourglassViTL16(HourglassViT):
    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(
            patch_size=16,
            num_layers=24,
            num_heads=16,
            hidden_dim=1024,
            mlp_dim=4096,
            **kwargs,
        )


class HourglassViTH14(HourglassViT):
    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(
            patch_size=14,
            num_layers=32,
            num_heads=16,
            hidden_dim=1280,
            mlp_dim=5120,
            **kwargs,
        )

