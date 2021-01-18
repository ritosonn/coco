from typing import NamedTuple
import torch
import torch.nn as nn
import torch.nn.functional as F

class FCOSConfig(NamedTuple):
    img_h: int
    img_w: int
    x8_ch: int
    x16_ch: int
    x32_ch: int
    fpn_ch: int

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

def test_fpn():
    from backbone import EfficientNetBackbone

    b = EfficientNetBackbone("tf_efficientnet_b0_ns")
    print(b)
    f = FeaturePyramid(FCOSConfig(800, 1024, 40, 112, 320, 256))
    print(f)
    x = torch.zeros((1, 3, 800, 1024))
    print(x.shape)
    x8, x16, x32 = b(x)
    print(x8.shape, x16.shape, x32.shape)
    o8, o16, o32, o64, o128 = f(x8, x16, x32)
    print(o8.shape, o16.shape, o32.shape, o64.shape, o128.shape)

if __name__ == "__main__":
    test_fpn()