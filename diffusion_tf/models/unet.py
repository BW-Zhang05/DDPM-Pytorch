import torch
from torch import nn
import torch.nn.functional as F

from .. import nn as nn_utils


def swish(x):
    return x * torch.sigmoid(x)


def norm_layer(channels: int) -> nn.GroupNorm:
    groups = 32 if channels >= 32 else max(1, channels // 4)
    return nn.GroupNorm(num_groups=groups, num_channels=channels)


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, temb_ch, dropout, use_conv_shortcut=False):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.use_conv_shortcut = use_conv_shortcut

        self.norm1 = norm_layer(in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.temb_proj = nn.Linear(temb_ch, out_ch)
        self.norm2 = norm_layer(out_ch)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)

        if in_ch != out_ch:
            if use_conv_shortcut:
                self.shortcut = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
            else:
                self.shortcut = nn.Conv2d(in_ch, out_ch, kernel_size=1)
        else:
            self.shortcut = None

    def forward(self, x, temb):
        h = self.norm1(x)
        h = swish(h)
        h = self.conv1(h)
        h = h + self.temb_proj(swish(temb))[:, :, None, None]
        h = self.norm2(h)
        h = swish(h)
        h = self.dropout(h)
        h = self.conv2(h)
        if self.shortcut is not None:
            x = self.shortcut(x)
        return x + h


class AttnBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = norm_layer(channels)
        self.q = nn.Conv2d(channels, channels, kernel_size=1)
        self.k = nn.Conv2d(channels, channels, kernel_size=1)
        self.v = nn.Conv2d(channels, channels, kernel_size=1)
        self.proj_out = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x)
        q = self.q(h).reshape(B, C, H * W)
        k = self.k(h).reshape(B, C, H * W)
        v = self.v(h).reshape(B, C, H * W)

        w = torch.einsum('bct,bcs->bts', q, k) * (C ** -0.5) # bct * bcs = bts
        w = torch.softmax(w, dim=-1)
        h = torch.einsum('bts,bcs->bct', w, v)
        h = h.reshape(B, C, H, W)
        h = self.proj_out(h)
        return x + h


class Downsample(nn.Module):
    def __init__(self, channels, with_conv=True):
        super().__init__()
        self.with_conv = with_conv
        if with_conv:
            self.op = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)
        else:
            self.op = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        return self.op(x)


class Upsample(nn.Module):
    def __init__(self, channels, with_conv=True):
        super().__init__()
        self.with_conv = with_conv
        if with_conv:
            self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        else:
            self.conv = None

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.0, mode='nearest')
        if self.conv is not None:
            x = self.conv(x)
        return x


class UNet(nn.Module):
    def __init__(
        self,
        *,
        in_ch,
        out_ch,
        ch,
        ch_mult=(1, 2, 4, 8),
        num_res_blocks=2,
        attn_resolutions=(16,),
        dropout=0.0,
        resamp_with_conv=True,
        image_size=32,
    ):
        super().__init__()
        self.ch = ch
        self.temb_ch = ch * 4
        self.num_resolutions = len(ch_mult)
        self.attn_resolutions = set(attn_resolutions)

        self.temb = nn.Sequential(
            nn.Linear(ch, self.temb_ch),
            nn.SiLU(),
            nn.Linear(self.temb_ch, self.temb_ch),
        )

        self.conv_in = nn.Conv2d(in_ch, ch, kernel_size=3, padding=1) # 通道扩大到ch,分辨率不变

        self.down = nn.ModuleList()
        self.up = nn.ModuleList()
        chs = [ch]
        curr_res = image_size

        # Downsampling
        for i_level, mult in enumerate(ch_mult): # ch_mult=(1, 2, 4, 8)
            block = nn.ModuleList()
            attn = nn.ModuleList()
            out_ch_level = ch * mult
            for _ in range(num_res_blocks):
                block.append(ResBlock(chs[-1], out_ch_level, self.temb_ch, dropout))
                chs.append(out_ch_level)
                if curr_res in self.attn_resolutions:
                    attn.append(AttnBlock(out_ch_level))
                else:
                    attn.append(None)
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(out_ch_level, with_conv=resamp_with_conv)
                chs.append(out_ch_level)
                curr_res //= 2
            else:
                down.downsample = None
            self.down.append(down)

        # Middle
        self.mid_block1 = ResBlock(chs[-1], chs[-1], self.temb_ch, dropout)
        self.mid_attn = AttnBlock(chs[-1])
        self.mid_block2 = ResBlock(chs[-1], chs[-1], self.temb_ch, dropout)

        # Upsampling
        curr_ch = chs[-1]
        for i_level, mult in reversed(list(enumerate(ch_mult))):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            out_ch_level = ch * mult
            for _ in range(num_res_blocks + 1):
                skip_ch = chs.pop()
                block.append(ResBlock(curr_ch + skip_ch, out_ch_level, self.temb_ch, dropout))
                curr_ch = out_ch_level
                if curr_res in self.attn_resolutions:
                    attn.append(AttnBlock(out_ch_level))
                else:
                    attn.append(None)
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(out_ch_level, with_conv=resamp_with_conv)
                curr_res *= 2
            else:
                up.upsample = None
            self.up.append(up)

        self.norm_out = norm_layer(ch)
        self.conv_out = nn.Conv2d(ch, out_ch, kernel_size=3, padding=1)

    def forward(self, x, t): # 输入图像 x 与
        temb = nn_utils.get_timestep_embedding(t, self.ch) # self.ch纬度大小 e.g. 128
        temb = self.temb(temb) # 映射到4*ch

        hs = []
        h = self.conv_in(x) # 通道扩大到ch,分辨率不变
        hs.append(h)

        for down in self.down:
            for block, attn in zip(down.block, down.attn):
                h = block(h, temb)
                if attn is not None:
                    h = attn(h)
                hs.append(h)
            if down.downsample is not None:
                h = down.downsample(h)
                hs.append(h)

        h = self.mid_block1(h, temb)
        h = self.mid_attn(h)
        h = self.mid_block2(h, temb)

        for up in self.up:
            for block, attn in zip(up.block, up.attn):
                h = torch.cat([h, hs.pop()], dim=1)
                h = block(h, temb)
                if attn is not None:
                    h = attn(h)
            if up.upsample is not None:
                h = up.upsample(h)

        h = self.norm_out(h)
        h = swish(h)
        h = self.conv_out(h)
        return h


__all__ = ["UNet"]
