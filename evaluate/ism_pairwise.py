"""Pairwise ISM & FDFR between two directories of corresponding face images."""

from deepface import DeepFace
import numpy as np
import os
import torch
import torch.nn.functional as F
import argparse
import re


def extract_id(filename):
    """Extract numeric ID from various filename patterns.
    e.g. '10372.jpg' -> '10372', '5_noise_10372_output.png' -> '10372',
         'adversarial_10372_output.png' -> '10372', '10372_output.png' -> '10372'
    """
    stem = os.path.splitext(filename)[0]
    # remove _output suffix
    if stem.endswith("_output"):
        stem = stem[: -len("_output")]
    # try known prefix patterns: "adversarial_", "{digit}_noise_", etc.
    # general approach: take the last purely numeric segment
    parts = stem.split("_")
    for part in reversed(parts):
        if part.isdigit():
            return part
    return stem


def compute_face_embedding(img_path):
    """Extract face embedding of the largest face in the image."""
    try:
        resps = DeepFace.represent(
            img_path=img_path,
            model_name="ArcFace",
            enforce_detection=True,
            detector_backend="retinaface",
            align=True,
        )
        resps = list(resps)
        resps.sort(
            key=lambda r: r["facial_area"]["h"] * r["facial_area"]["w"],
            reverse=True,
        )
        return np.array(resps[0]["embedding"])
    except Exception:
        return None


def find_matching_file(stem, tgt_dir):
    """Find a file in tgt_dir whose stem matches (e.g. '10372' matches '10372.jpg', '10372_output.png')."""
    for f in os.listdir(tgt_dir):
        tgt_stem = os.path.splitext(f)[0]
        if tgt_stem == stem or tgt_stem == f"{stem}_output":
            return os.path.join(tgt_dir, f)
    return None


def pairwise_ism(src_dir, tgt_dir):
    """Compute pairwise cosine similarity between corresponding images."""
    src_files = sorted(
        f for f in os.listdir(src_dir) if f.endswith((".jpg", ".png"))
    )

    # pre-build target lookup: extract numeric id from filenames
    tgt_lookup = {}
    for f in os.listdir(tgt_dir):
        if not f.endswith((".jpg", ".png")):
            continue
        fid = extract_id(f)
        tgt_lookup[fid] = os.path.join(tgt_dir, f)

    results = []
    fail_count = 0
    total = len(src_files)

    for src_name in src_files:
        stem = extract_id(src_name)
        tgt_path = tgt_lookup.get(stem)

        if tgt_path is None:
            print(f"[SKIP] no match for {src_name}")
            fail_count += 1
            continue

        src_emb = compute_face_embedding(os.path.join(src_dir, src_name))
        tgt_emb = compute_face_embedding(tgt_path)

        if src_emb is None or tgt_emb is None:
            label = "src" if src_emb is None else "tgt"
            print(f"[FAIL] face detection failed ({label}): {src_name}")
            fail_count += 1
            continue

        sim = F.cosine_similarity(
            torch.tensor(src_emb, dtype=torch.float32),
            torch.tensor(tgt_emb, dtype=torch.float32),
            dim=0,
        ).item()
        results.append((src_name, sim))
        print(f"[OK] {src_name:>20s}  ISM = {sim:.4f}")

    avg_ism = np.mean([r[1] for r in results]) if results else 0.0
    fdfr = fail_count / total if total > 0 else 1.0

    print("\n" + "=" * 50)
    print(f"Total pairs : {total}")
    print(f"Matched     : {len(results)}")
    print(f"Failed      : {fail_count}")
    print(f"Avg ISM     : {avg_ism:.4f}")
    print(f"FDFR        : {fdfr:.4f}")
    return avg_ism, fdfr


def main():
    parser = argparse.ArgumentParser(description="Pairwise ISM & FDFR evaluation")
    parser.add_argument("--src_dir", type=str, required=True, help="source (clean) image dir")
    parser.add_argument("--tgt_dir", type=str, required=True, help="target (output) image dir")
    args = parser.parse_args()
    pairwise_ism(args.src_dir, args.tgt_dir)


if __name__ == "__main__":
    main()
