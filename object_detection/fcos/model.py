from typing import NamedTuple
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from backbone import EfficientNetBackbone

class FCOSConfig(NamedTuple):
    backbone_name: str
    img_h: int
    img_w: int
    x8_ch: int
    x16_ch: int
    x32_ch: int
    fpn_ch: int
    n_cls: int

class EfficientNetBackbone(nn.Module):
    def __init__(self, model_name: str):
        super().__init__()
        model = timm.create_model(model_name, pretrained=True)
        self.backbone_8 = nn.Sequential(
            model.conv_stem,
            model.bn1,
            model.act1,
            model.blocks[0],
            model.blocks[1],
            model.blocks[2],
        )
        self.backbone_16 = nn.Sequential(
            model.blocks[3],
            model.blocks[4],
        )
        self.backbone_32 = nn.Sequential(
            model.blocks[5],
            model.blocks[6],
        )
    def forward(self, x):
        out_8 = self.backbone_8(x)
        out_16 = self.backbone_16(out_8)
        out_32 = self.backbone_32(out_16)
        return out_8, out_16, out_32

class FeaturePyramid(nn.Module):
    def __init__(self, cfg: FCOSConfig):
        super().__init__()

        self.p3_in = nn.Conv2d(cfg.x8_ch, cfg.fpn_ch, kernel_size=1, stride=1, padding=0)
        self.p4_in = nn.Conv2d(cfg.x16_ch, cfg.fpn_ch, kernel_size=1, stride=1, padding=0)
        self.p5_in = nn.Conv2d(cfg.x32_ch, cfg.fpn_ch, kernel_size=1, stride=1, padding=0)

        self.p3_out = nn.Conv2d(cfg.fpn_ch, cfg.fpn_ch, kernel_size=3, stride=1, padding=1)
        self.p4_out = nn.Conv2d(cfg.fpn_ch, cfg.fpn_ch, kernel_size=3, stride=1, padding=1)
        self.p5_out = nn.Conv2d(cfg.fpn_ch, cfg.fpn_ch, kernel_size=3, stride=1, padding=1)
        self.upsample_16 = lambda y_32: F.interpolate(y_32, size=(cfg.img_h // 16, cfg.img_w // 16), mode="nearest")
        self.upsample_8 = lambda y_16: F.interpolate(y_16, size=(cfg.img_h // 8, cfg.img_w // 8), mode="nearest")

        self.p6 = nn.Conv2d(cfg.fpn_ch, cfg.fpn_ch, kernel_size=3, stride=2, padding=1)
        self.p7 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(cfg.fpn_ch, cfg.fpn_ch, kernel_size=3, stride=2, padding=1)
        )

    def forward(self, x_8, x_16, x_32):
        # adjust ch
        y_8 = self.p3_in(x_8)
        y_16 = self.p4_in(x_16)
        y_32 = self.p5_in(x_32)

        # upsample
        out_32 = self.p5_out(y_32)
        out_16 = self.p4_out(y_16 + self.upsample_16(y_32))
        out_8 = self.p3_out(y_8 + self.upsample_8(y_16))

        # downsample
        out_64 = self.p6(out_32)
        out_128 = self.p7(out_64)

        return out_8, out_16, out_32, out_64, out_128

class FCOSHead(nn.Module):
    def __init__(self, cfg: FCOSConfig, downsample: int):
        super().__init__()
        self.regression_net = self._generate_conv_block(cfg.fpn_ch, 4)
        self.regression_head = nn.Sequential(
            nn.Conv2d(cfg.fpn_ch, 4, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )
        self.classification_net = self._generate_conv_block(cfg.fpn_ch, 4)
        self.classification_head = nn.Conv2d(cfg.fpn_ch, cfg.n_cls, kernel_size=3, stride=1, padding=1)
        self.centerness_head = nn.Conv2d(cfg.fpn_ch, 1, kernel_size=3, stride=1, padding=1)
        self.downsample = downsample

    @staticmethod
    def _generate_conv_block(c: int, rep: int):
        ret = []
        for _ in range(rep):
            ret += [
                nn.Conv2d(c, c, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(c),
                nn.ReLU(inplace=True),
            ]
        return nn.Sequential(*ret)

    def forward(self, x):
        y_reg = self.regression_net(x)
        y_cls = self.classification_net(x)
        out_reg = self.regression_head(y_reg) * self.downsample  # scaling
        out_cls = self.classification_head(y_cls)
        out_center = self.centerness_head(y_cls)
        return out_reg, out_cls, out_center


class FCOS(nn.Module):
    def __init__(self, cfg: FCOSConfig):
        super().__init__()
        self.backbone = EfficientNetBackbone(cfg.backbone_name)
        self.fpn = FeaturePyramid(cfg)
        self.head_8 = FCOSHead(cfg, 8)
        self.head_16 = FCOSHead(cfg, 16)
        self.head_32 = FCOSHead(cfg, 32)
        self.head_64 = FCOSHead(cfg, 64)
        self.head_128 = FCOSHead(cfg, 128)

    def forward(self, x):
        x8, x16, x32 = self.backbone(x)
        y8, y16, y32, y64, y128 = self.fpn(x8, x16, x32)
        o8 = self.head_8(y8)
        o16 = self.head_16(y16)
        o32 = self.head_32(y32)
        o64 = self.head_64(y64)
        o128 = self.head_128(y128)
        return o8, o16, o32, o64, o128

def check():

    cfg = FCOSConfig("tf_efficientnet_b0_ns", 800, 1024, 40, 112, 320, 256, 80)
    model = FCOS(cfg)
    print(model)
    x = torch.zeros((1, 3, 800, 1024))
    print(x.shape)
    o8, o16, o32, o64, o128 = model(x)
    o_reg, o_cls, o_center = o8
    print(o_reg.shape, o_cls.shape, o_center.shape)

if __name__ == "__main__":
    check()