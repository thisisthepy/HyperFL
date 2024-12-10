from functools import partial
import torch
import torch.nn as nn
import timm.models.vision_transformer


class Adapter(nn.Module):
    def __init__(self, input_dim=384, reduction_factor=16):
        super().__init__()
        self.input_dim = input_dim
        self.down_sample_size = self.input_dim // reduction_factor
        self.down_sampler = nn.Linear(self.input_dim, self.down_sample_size)
        self.activation = nn.ReLU()
        self.up_sampler = nn.Linear(self.down_sample_size, self.input_dim)
        self.norm = nn.LayerNorm(self.input_dim)

    def forward(self, x):
        z = self.down_sampler(x)
        z = self.activation(z)
        z = self.up_sampler(z)
        z = self.norm(z)
        x = x + z
        return x


class Block(timm.models.vision_transformer.Block):
    def __init__(self, **kwargs):
        super(Block, self).__init__(**kwargs)

        self.adapter = Adapter(input_dim=kwargs['dim'])

    def forward(self, x):
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        x = x + self.adapter(self.drop_path2(self.ls2(self.mlp(self.norm2(x)))))
        return x


class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    def __init__(self, **kwargs):
        super(VisionTransformer, self).__init__(**kwargs)

        self.blocks = nn.Sequential(*[Block(dim=kwargs['embed_dim'], num_heads=kwargs['num_heads']) for i in range(12)])


def vit_small_patch16(**kwargs):
    model = VisionTransformer(
        img_size=384, patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model
