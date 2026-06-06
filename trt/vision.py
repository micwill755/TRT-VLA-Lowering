import copy

import torch
import torch.nn as nn

class GROOTVisualEmbed(nn.Module):
    def __init__(self, groot):
        super().__init__()
        self.eagle_model = groot.backbone.eagle_model

    def forward(self, pixel_values):
        return self.eagle_model.extract_feature(pixel_values)

class PI05VisualEmbed(nn.Module):
    def __init__(self, core):
        super().__init__()
        self.paligemma_with_expert = core.paligemma_with_expert

    def forward(self, image):
        return self.paligemma_with_expert.embed_image(image)

class FP16CastWrapper(nn.Module):
    def __init__(self, trt_model):
        super().__init__()
        self.trt_model = trt_model

    def forward(self, image):
        return self.trt_model(image.to(torch.float16))

class SmolVLAVisualEmbed(nn.Module):
    def __init__(self, core):
        super().__init__()
        self.vlm_with_expert = core.vlm_with_expert

    def forward(self, image):
        return self.vlm_with_expert.embed_image(image)
