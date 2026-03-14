"""
Visualize UNet architecture with torchview.

Example:
python scripts/visualize_unet.py --out_dir visualizations --image_size 32 --ch 128
"""

import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import fire
import torch
from torchview import draw_graph

from diffusion_tf.models.unet import UNet


def main(
    out_dir="visualizations",
    filename="unet",
    image_size=32,
    in_ch=3,
    out_ch=3,
    ch=128,
    ch_mult=(1, 2, 2, 2),
    num_res_blocks=2,
    attn_resolutions=(16,),
    dropout=0.0,
    resamp_with_conv=True,
    batch_size=1,
    device="cpu",
):
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device(device)

    model = UNet(
        in_ch=in_ch,
        out_ch=out_ch,
        ch=ch,
        ch_mult=tuple(ch_mult),
        num_res_blocks=num_res_blocks,
        attn_resolutions=tuple(attn_resolutions),
        dropout=dropout,
        resamp_with_conv=resamp_with_conv,
        image_size=image_size,
    ).to(device)
    model.eval()

    x = torch.zeros((batch_size, in_ch, image_size, image_size), device=device)
    t = torch.zeros((batch_size,), dtype=torch.long, device=device)

    graph = draw_graph(
        model,
        input_data=(x, t),
        expand_nested=True,
        graph_name="UNet",
    )
    out_path = os.path.join(out_dir, filename)
    graph.visual_graph.render(out_path, format="png", cleanup=True)
    print(f"Saved: {out_path}.png")


if __name__ == "__main__":
    fire.Fire(main)
