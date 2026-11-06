"""Compute CLIP-IQA scores for images in a directory using piq."""

import argparse
import os
import torch
from PIL import Image
from torchvision import transforms
from piq import CLIPIQA


def main():
    parser = argparse.ArgumentParser(description="Compute CLIP-IQA for images in a directory")
    parser.add_argument("--img_dir", type=str, required=True, help="image directory")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    clipiqa = CLIPIQA().to(args.device)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    img_files = sorted(
        f for f in os.listdir(args.img_dir) if f.endswith((".jpg", ".png"))
    )

    scores = []
    for f in img_files:
        img = Image.open(os.path.join(args.img_dir, f)).convert("RGB")
        x = transform(img).unsqueeze(0).to(args.device)
        with torch.no_grad():
            score = clipiqa(x).item()
        scores.append(score)
        print(f"{f:>40s}  CLIP-IQA = {score:.4f}")

    print(f"\n{'=' * 50}")
    print(f"Images       : {len(scores)}")
    print(f"Avg CLIP-IQA : {sum(scores) / len(scores):.4f}")


if __name__ == "__main__":
    main()
