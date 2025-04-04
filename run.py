import sys
import os
import math
import itertools
import urllib.request
import argparse
from functools import partial
from PIL import Image
import matplotlib
import torch
import torch.nn.functional as F
from torchvision import transforms
import mmcv
from mmcv.runner import load_checkpoint
from dinov2.eval.depth.models import build_depther


class CenterPadding(torch.nn.Module):
    def __init__(self, multiple):
        super().__init__()
        self.multiple = multiple

    def _get_pad(self, size):
        new_size = math.ceil(size / self.multiple) * self.multiple
        pad_size = new_size - size
        pad_size_left = pad_size // 2
        pad_size_right = pad_size - pad_size_left
        return pad_size_left, pad_size_right

    @torch.inference_mode()
    def forward(self, x):
        pads = list(itertools.chain.from_iterable(self._get_pad(m) for m in x.shape[:1:-1]))
        output = F.pad(x, pads)
        return output


def create_depther(cfg, backbone_model, backbone_size, head_type):
    train_cfg = cfg.get("train_cfg")
    test_cfg = cfg.get("test_cfg")
    depther = build_depther(cfg.model, train_cfg=train_cfg, test_cfg=test_cfg)

    depther.backbone.forward = partial(
        backbone_model.get_intermediate_layers,
        n=cfg.model.backbone.out_indices,
        reshape=True,
        return_class_token=cfg.model.backbone.output_cls_token,
        norm=cfg.model.backbone.final_norm,
    )

    if hasattr(backbone_model, "patch_size"):
        depther.backbone.register_forward_pre_hook(lambda _, x: CenterPadding(backbone_model.patch_size)(x[0]))

    return depther


def load_image_from_url(url: str) -> Image:
    with urllib.request.urlopen(url) as f:
        return Image.open(f).convert("RGB")


def make_depth_transform() -> transforms.Compose:
    return transforms.Compose([
        transforms.ToTensor(),
        lambda x: 255.0 * x[:3],
        transforms.Normalize(
            mean=(123.675, 116.28, 103.53),
            std=(58.395, 57.12, 57.375),
        ),
    ])


def render_depth(values, colormap_name="magma_r") -> Image:
    min_value, max_value = values.min(), values.max()
    normalized_values = (values - min_value) / (max_value - min_value)
    colormap = matplotlib.colormaps[colormap_name]
    colors = colormap(normalized_values, bytes=True)
    colors = colors[:, :, :3]
    return Image.fromarray(colors)


def process_image(image_path, model, transform):
    image = Image.open(image_path).convert("RGB")
    rescaled_image = image.resize((image.width, image.height))
    transformed_image = transform(rescaled_image)
    batch = transformed_image.unsqueeze(0).cuda()
    print(f"Processing {image_path}...")
    with torch.inference_mode():
        result = model.whole_inference(batch, img_meta=None, rescale=True)
    depth_image = render_depth(result.squeeze().cpu())
    return depth_image


def prompt_choice(prompt, options):
    print(prompt)
    for i, option in enumerate(options, 1):
        print(f"{i}. {option}")
    while True:
        try:
            choice = int(input("Enter your choice (number): "))
            if 1 <= choice <= len(options):
                return options[choice - 1]
        except ValueError:
            pass
        print("Invalid input. Please enter a valid number.")


def main():
    parser = argparse.ArgumentParser(description="DINOv2 Depth Estimation")
    parser.add_argument("--input", type=str, required=True, help="Input folder containing .png images")
    parser.add_argument("--output", type=str, required=True, help="Output folder to save depth images")
    args = parser.parse_args()

    BACKBONE_SIZE = prompt_choice("Select Backbone Size:", ["small", "base", "large", "giant"])
    HEAD_DATASET = prompt_choice("Select Dataset:", ["nyu", "kitti", "cityscapes"])
    HEAD_TYPE = prompt_choice("Select Head Type:", ["dpt", "deeplabv3"])

    input_folder = args.input
    output_folder = args.output

    backbone_archs = {
        "small": "vits14",
        "base": "vitb14",
        "large": "vitl14",
        "giant": "vitg14",
    }
    backbone_arch = backbone_archs[BACKBONE_SIZE]
    backbone_name = f"dinov2_{backbone_arch}"

    print("\nLoading DINOv2 backbone from torch.hub...")
    backbone_model = torch.hub.load(repo_or_dir="facebookresearch/dinov2", model=backbone_name)
    backbone_model.eval().cuda()

    DINOV2_BASE_URL = "https://dl.fbaipublicfiles.com/dinov2"
    head_config_url = f"{DINOV2_BASE_URL}/{backbone_name}/{backbone_name}_{HEAD_DATASET}_{HEAD_TYPE}_config.py"
    head_checkpoint_url = f"{DINOV2_BASE_URL}/{backbone_name}/{backbone_name}_{HEAD_DATASET}_{HEAD_TYPE}_head.pth"

    print("Downloading depth head config...")
    cfg_str = urllib.request.urlopen(head_config_url).read().decode()
    cfg = mmcv.Config.fromstring(cfg_str, file_format=".py")

    print("Creating depth model...")
    model = create_depther(cfg, backbone_model, BACKBONE_SIZE, HEAD_TYPE)
    load_checkpoint(model, head_checkpoint_url, map_location="cpu")
    model.eval().cuda()

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    transform = make_depth_transform()
    image_files = [f for f in os.listdir(input_folder) if f.endswith(".png")]

    total = len(image_files)
    for idx, image_filename in enumerate(image_files, 1):
        image_path = os.path.join(input_folder, image_filename)
        depth_image = process_image(image_path, model, transform)
        output_image_path = os.path.join(output_folder, image_filename)
        depth_image.save(output_image_path)
        print(f"[{idx}/{total}] Saved depth image as {output_image_path}")


if __name__ == "__main__":
    main()
