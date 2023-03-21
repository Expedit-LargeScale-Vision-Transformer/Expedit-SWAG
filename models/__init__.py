# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from enum import Enum
from typing import Any, Optional

from torch.hub import load_state_dict_from_url

from .regnet import RegNetY32gf, RegNetY16gf, RegNetY128gf
from .vision_transformer import ViTB16, ViTH14, ViTL16
from .hourglass_vit import HourglassViTH14, HourglassViTL16
from .evit import EViTH14, EViTL16, EViTB16
from .act_vision_transformer import ActViTH14, ActViTL16
from .smyrf_vision_transformer import SmyrfViTH14, SmyrfViTL16
import tome
from .ats_vision_trainsformer import ATSViTL16, ATSViTH14
from .evo_vit import EvoViTL16, EvoViTH14


class ModelCheckpoints(Enum):
    regnety_16gf = "https://dl.fbaipublicfiles.com/SWAG/regnety_16gf.torch"
    regnety_16gf_in1k = "https://dl.fbaipublicfiles.com/SWAG/regnety_16gf_in1k.torch"
    regnety_32gf = "https://dl.fbaipublicfiles.com/SWAG/regnety_32gf.torch"
    regnety_32gf_in1k = "https://dl.fbaipublicfiles.com/SWAG/regnety_32gf_in1k.torch"
    regnety_128gf = "https://dl.fbaipublicfiles.com/SWAG/regnety_128gf.torch"
    regnety_128gf_in1k = "https://dl.fbaipublicfiles.com/SWAG/regnety_128gf_in1k.torch"
    vit_b16 = "https://dl.fbaipublicfiles.com/SWAG/vit_b16.torch"
    vit_b16_in1k = "https://dl.fbaipublicfiles.com/SWAG/vit_b16_in1k.torch"
    vit_l16 = "https://dl.fbaipublicfiles.com/SWAG/vit_l16.torch"
    vit_l16_in1k = "https://dl.fbaipublicfiles.com/SWAG/vit_l16_in1k.torch"
    vit_h14 = "https://dl.fbaipublicfiles.com/SWAG/vit_h14.torch"
    vit_h14_in1k = "https://dl.fbaipublicfiles.com/SWAG/vit_h14_in1k.torch"

IN1K_CLASSES = 1000


def build_model(
    cls: type,
    checkpoint_path: str,
    pretrained: bool = True,
    progress: bool = True,
    **kwargs
):
    model = cls(**kwargs)
    if pretrained:
        checkpoint = load_state_dict_from_url(checkpoint_path, progress=progress)
        model.load_state_dict(checkpoint)
    return model


def regnety_16gf(pretrained: bool = True, progress: bool = True, **kwargs):
    return build_model(
        RegNetY16gf,
        ModelCheckpoints.regnety_16gf.value,
        num_classes=None,
        pretrained=pretrained,
        progress=progress,
        **kwargs
    )


def regnety_16gf_in1k(pretrained: bool = True, progress: bool = True, **kwargs):
    return build_model(
        RegNetY16gf,
        ModelCheckpoints.regnety_16gf_in1k.value,
        num_classes=IN1K_CLASSES,
        pretrained=pretrained,
        progress=progress,
        **kwargs
    )


def regnety_32gf(pretrained: bool = True, progress: bool = True, **kwargs):
    return build_model(
        RegNetY32gf,
        ModelCheckpoints.regnety_32gf.value,
        num_classes=None,
        pretrained=pretrained,
        progress=progress,
        **kwargs
    )


def regnety_32gf_in1k(pretrained: bool = True, progress: bool = True, **kwargs):
    return build_model(
        RegNetY32gf,
        ModelCheckpoints.regnety_32gf_in1k.value,
        num_classes=IN1K_CLASSES,
        pretrained=pretrained,
        progress=progress,
        **kwargs
    )


def regnety_128gf(pretrained: bool = True, progress: bool = True, **kwargs):
    return build_model(
        RegNetY128gf,
        ModelCheckpoints.regnety_128gf.value,
        num_classes=None,
        pretrained=pretrained,
        progress=progress,
        **kwargs
    )


def regnety_128gf_in1k(pretrained: bool = True, progress: bool = True, **kwargs):
    return build_model(
        RegNetY128gf,
        ModelCheckpoints.regnety_128gf_in1k.value,
        num_classes=IN1K_CLASSES,
        pretrained=pretrained,
        progress=progress,
        **kwargs
    )


def vit_b16(pretrained: bool = True, progress: bool = True, **kwargs):
    return build_model(
        ViTB16,
        ModelCheckpoints.vit_b16.value,
        num_classes=None,
        pretrained=pretrained,
        progress=progress,
        **kwargs
    )


def vit_b16_in1k(pretrained: bool = True, progress: bool = True, **kwargs):
    return build_model(
        ViTB16,
        ModelCheckpoints.vit_b16_in1k.value,
        image_size=384,
        num_classes=IN1K_CLASSES,
        pretrained=pretrained,
        progress=progress,
        **kwargs
    )


def vit_l16(pretrained: bool = True, progress: bool = True, **kwargs):
    return build_model(
        ViTL16,
        ModelCheckpoints.vit_l16.value,
        num_classes=None,
        pretrained=pretrained,
        progress=progress,
        **kwargs
    )


def vit_l16_in1k(pretrained: bool = True, progress: bool = True, **kwargs):
    return build_model(
        ViTL16,
        ModelCheckpoints.vit_l16_in1k.value,
        # image_size=512,
        num_classes=IN1K_CLASSES,
        pretrained=pretrained,
        progress=progress,
        **kwargs
    )


def vit_h14(pretrained: bool = True, progress: bool = True, **kwargs):
    return build_model(
        ViTH14,
        ModelCheckpoints.vit_h14.value,
        num_classes=None,
        pretrained=pretrained,
        progress=progress,
        **kwargs
    )


def vit_h14_in1k(pretrained: bool = True, progress: bool = True, **kwargs):
    return build_model(
        ViTH14,
        ModelCheckpoints.vit_h14_in1k.value,
        # image_size=518,
        num_classes=IN1K_CLASSES,
        pretrained=pretrained,
        progress=progress,
        **kwargs
    )

def hourglass_vit_l16_in1k(pretrained: bool = True, progress: bool = True, **kwargs):
    return build_model(
        HourglassViTL16,
        ModelCheckpoints.vit_l16_in1k.value,
        # image_size=512,
        num_classes=IN1K_CLASSES,
        pretrained=pretrained,
        progress=progress,
        cluster_after_output=False,
        **kwargs
    )

def hourglass_vit_h14_in1k(pretrained: bool = True, progress: bool = True, **kwargs):
    return build_model(
        HourglassViTH14,
        ModelCheckpoints.vit_h14_in1k.value,
        # image_size=518,
        num_classes=IN1K_CLASSES,
        pretrained=pretrained,
        progress=progress,
        cluster_after_output=False,
        **kwargs
    )

def tome_vit_l16_in1k(pretrained: bool = True, progress: bool = True, tome_r = 0, **kwargs):
    model = vit_l16_in1k(pretrained, progress, **kwargs)
    tome.patch.swag(model)
    model.r = tome_r
    return model

def tome_vit_h14_in1k(pretrained: bool = True, progress: bool = True, tome_r = 0, **kwargs):
    model = vit_h14_in1k(pretrained, progress, **kwargs)
    tome.patch.swag(model)
    model.r = tome_r
    return model

def e_vit_b16_in1k(pretrained: bool = True, progress: bool = True, **kwargs):
    return build_model(
        EViTB16,
        ModelCheckpoints.vit_b16_in1k.value,
        # image_size=384,
        num_classes=IN1K_CLASSES,
        pretrained=pretrained,
        progress=progress,
        **kwargs
    )


def e_vit_l16_in1k(pretrained: bool = True, progress: bool = True, **kwargs):
    return build_model(
        EViTL16,
        ModelCheckpoints.vit_l16_in1k.value,
        # image_size=512,
        num_classes=IN1K_CLASSES,
        pretrained=pretrained,
        progress=progress,
        **kwargs
    )


def e_vit_h14_in1k(pretrained: bool = True, progress: bool = True, **kwargs):
    return build_model(
        EViTH14,
        ModelCheckpoints.vit_h14_in1k.value,
        # image_size=518,
        num_classes=IN1K_CLASSES,
        pretrained=pretrained,
        progress=progress,
        **kwargs
    )

def act_vit_l16_in1k(pretrained: bool = True, progress: bool = True, **kwargs):
    return build_model(
        ActViTL16,
        ModelCheckpoints.vit_l16_in1k.value,
        # image_size=512,
        num_classes=IN1K_CLASSES,
        pretrained=pretrained,
        progress=progress,
        **kwargs
    )

def act_vit_h14_in1k(pretrained: bool = True, progress: bool = True, **kwargs):
    return build_model(
        ActViTH14,
        ModelCheckpoints.vit_h14_in1k.value,
        # image_size=518,
        num_classes=IN1K_CLASSES,
        pretrained=pretrained,
        progress=progress,
        **kwargs
    )

def smyrf_vit_l16_in1k(pretrained: bool = True, progress: bool = True, **kwargs):
    return build_model(
        SmyrfViTL16,
        ModelCheckpoints.vit_l16_in1k.value,
        # image_size=512,
        num_classes=IN1K_CLASSES,
        pretrained=pretrained,
        progress=progress,
        **kwargs
    )

def smyrf_vit_h14_in1k(pretrained: bool = True, progress: bool = True, **kwargs):
    return build_model(
        SmyrfViTH14,
        ModelCheckpoints.vit_h14_in1k.value,
        # image_size=518,
        num_classes=IN1K_CLASSES,
        pretrained=pretrained,
        progress=progress,
        **kwargs
    )

def ats_vit_l16_in1k(pretrained: bool = True, progress: bool = True, **kwargs):
    return build_model(
        ATSViTL16,
        ModelCheckpoints.vit_l16_in1k.value,
        # image_size=512,
        num_classes=IN1K_CLASSES,
        pretrained=pretrained,
        progress=progress,
        **kwargs
    )

def ats_vit_h14_in1k(pretrained: bool = True, progress: bool = True, **kwargs):
    return build_model(
        ATSViTH14,
        ModelCheckpoints.vit_h14_in1k.value,
        # image_size=518,
        num_classes=IN1K_CLASSES,
        pretrained=pretrained,
        progress=progress,
        **kwargs
    )

def evo_vit_l16_in1k(pretrained: bool = True, progress: bool = True, **kwargs):
    return build_model(
        EvoViTL16,
        ModelCheckpoints.vit_l16_in1k.value,
        # image_size=512,
        num_classes=IN1K_CLASSES,
        pretrained=pretrained,
        progress=progress,
        **kwargs
    )

def evo_vit_h14_in1k(pretrained: bool = True, progress: bool = True, **kwargs):
    return build_model(
        EvoViTH14,
        ModelCheckpoints.vit_h14_in1k.value,
        # image_size=518,
        num_classes=IN1K_CLASSES,
        pretrained=pretrained,
        progress=progress,
        **kwargs
    )
