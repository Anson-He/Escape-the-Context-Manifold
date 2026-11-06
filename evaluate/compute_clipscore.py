"""Compute CLIP score between images in a directory and a given text prompt."""

import argparse
import os
import torch
import clip
from PIL import Image
import numpy as np


def compute_clip_score(img_dir, prompt, device="cuda", batch_size=8):
    model, preprocess = clip.load("ViT-B/32", device=device)

    text_tokens = clip.tokenize([prompt]).to(device)
    with torch.no_grad():
        text_feat = model.encode_text(text_tokens)
        text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)

    img_files = sorted(
        f for f in os.listdir(img_dir) if f.endswith((".jpg", ".png"))
    )

    scores = []
    for i in range(0, len(img_files), batch_size):
        batch_files = img_files[i : i + batch_size]
        images = []
        for f in batch_files:
            img = Image.open(os.path.join(img_dir, f)).convert("RGB")
            images.append(preprocess(img))
        image_input = torch.stack(images).to(device)

        with torch.no_grad():
            image_feat = model.encode_image(image_input)
            image_feat = image_feat / image_feat.norm(dim=-1, keepdim=True)
            # CLIP score = 100 * cosine similarity
            sim = (image_feat @ text_feat.T).squeeze(-1) * 100.0
            scores.extend(sim.cpu().tolist())

        for f, s in zip(batch_files, sim.cpu().tolist()):
            print(f"{f:>40s}  CLIPScore = {s:.2f}")

    avg_score = np.mean(scores)
    print(f"\n{'=' * 50}")
    print(f"Prompt   : {prompt}")
    print(f"Images   : {len(scores)}")
    print(f"Avg CLIP Score: {avg_score:.4f}")
    return avg_score


def main():
    parser = argparse.ArgumentParser(description="Compute CLIP score for images against a prompt")
    parser.add_argument("--img_dir", type=str, required=True, help="image directory")
    parser.add_argument("--prompt", type=str, required=True, help="text prompt")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=8)
    args = parser.parse_args()
    compute_clip_score(args.img_dir, args.prompt, args.device, args.batch_size)


if __name__ == "__main__":
    main()
