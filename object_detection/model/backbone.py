import timm
import torch
import torch.nn as nn

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
