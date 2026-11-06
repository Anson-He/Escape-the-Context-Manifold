#!/usr/bin/env python
# coding=utf-8
"""
FLUX Kontext Adversarial Attack - Manifold Hijack (Hook-based)

Based on attack_decontext.py hook architecture, adds v_context velocity hijacking:
  L = lambda_attn * (1 - context_prop) + lambda_vel * MSE(v_ctx, v_ref)

Uses register_forward_hook on norm_q, norm_k, and to_v (or to_qkv for fused)
to collect Q, K, V without replacing attention processors.
"""

import argparse
import logging
import math
import os
from pathlib import Path
from typing import Dict, List
import random
import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm.auto import tqdm
from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
    FluxKontextPipeline,
    FluxTransformer2DModel,
)
from diffusers.models.embeddings import apply_rotary_emb
from transformers import CLIPTokenizer, CLIPTextModel, T5TokenizerFast, T5EncoderModel

logger = get_logger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Dataset
# ═══════════════════════════════════════════════════════════════════════════════

class AdversarialDataset(Dataset):
    def __init__(self, condition_images_dir, prompts, resolution=512):
        self.condition_images_dir = Path(condition_images_dir)
        self.prompts = prompts
        self.resolution = resolution
        self.cond_files = sorted([
            f for f in self.condition_images_dir.iterdir()
            if f.suffix.lower() in ['.jpg', '.png', '.jpeg']
        ])
        self.image_transforms = transforms.Compose([
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

    def __len__(self):
        return len(self.cond_files)

    def __getitem__(self, idx):
        cond_path = self.cond_files[idx]
        cond_image = Image.open(cond_path).convert('RGB')
        cond_image = self.image_transforms(cond_image)
        return {'condition_image': cond_image, 'cond_path': str(cond_path)}


# ═══════════════════════════════════════════════════════════════════════════════
# Prompt encoding
# ═══════════════════════════════════════════════════════════════════════════════

def encode_prompt_flux_kontext(text_encoders, tokenizers, prompt, max_sequence_length=512, device=None):
    prompt = [prompt] if isinstance(prompt, str) else prompt
    text_inputs_clip = tokenizers[0](
        prompt, padding="max_length", max_length=77, truncation=True, return_tensors="pt"
    )
    with torch.no_grad():
        pooled_prompt_embeds = text_encoders[0](text_inputs_clip.input_ids.to(device)).pooler_output
    text_inputs_t5 = tokenizers[1](
        prompt, padding="max_length", max_length=max_sequence_length, truncation=True, return_tensors="pt"
    )
    with torch.no_grad():
        prompt_embeds = text_encoders[1](text_inputs_t5.input_ids.to(device))[0]
    text_ids = torch.zeros(prompt_embeds.shape[1], 3).to(device=device, dtype=prompt_embeds.dtype)
    return prompt_embeds, pooled_prompt_embeds, text_ids


# ═══════════════════════════════════════════════════════════════════════════════
# Attention Hook — collects Q, K (after norm) and V (before RoPE)
# ═══════════════════════════════════════════════════════════════════════════════

class AttentionHook:
    def __init__(self):
        self.qkv_data = {}
        self.hooks = []

    def register_hooks(self, transformer, selected_layers=None):
        if selected_layers is None:
            selected_layers = {'single_blocks': list(range(0, 38))}

        def create_norm_hook(layer_name, proj_type):
            def hook_fn(module, input, output):
                if output is not None and hasattr(output, 'shape'):
                    self.qkv_data[f"{layer_name}_{proj_type}"] = output.clone()
            return hook_fn

        def create_v_hook(layer_name, is_fused):
            """Hook on to_v (unfused) or to_qkv (fused) to capture V."""
            def hook_fn(module, input, output):
                if output is not None and hasattr(output, 'shape'):
                    if is_fused:
                        # to_qkv output: [B, S, 3*H*D] → split into q, k, v
                        v = output.chunk(3, dim=-1)[2]
                    else:
                        v = output
                    self.qkv_data[f"{layer_name}_v"] = v.clone()
            return hook_fn

        # Double blocks
        for i in selected_layers.get('double_blocks', []):
            if i < len(transformer.transformer_blocks):
                block = transformer.transformer_blocks[i]
                if hasattr(block, 'attn'):
                    attn = block.attn
                    for proj_type, norm_layer in [('q', 'norm_q'), ('k', 'norm_k')]:
                        if hasattr(attn, norm_layer):
                            hook = getattr(attn, norm_layer).register_forward_hook(
                                create_norm_hook(f'double_{i}', proj_type)
                            )
                            self.hooks.append(hook)
                    # V hook
                    is_fused = getattr(attn, 'fused_projections', False)
                    if is_fused and hasattr(attn, 'to_qkv'):
                        hook = attn.to_qkv.register_forward_hook(
                            create_v_hook(f'double_{i}', is_fused=True)
                        )
                    elif hasattr(attn, 'to_v'):
                        hook = attn.to_v.register_forward_hook(
                            create_v_hook(f'double_{i}', is_fused=False)
                        )
                    else:
                        continue
                    self.hooks.append(hook)

        # Single blocks
        for i in selected_layers.get('single_blocks', []):
            if i < len(transformer.single_transformer_blocks):
                block = transformer.single_transformer_blocks[i]
                if hasattr(block, 'attn'):
                    attn = block.attn
                    for proj_type, norm_layer in [('q', 'norm_q'), ('k', 'norm_k')]:
                        if hasattr(attn, norm_layer):
                            hook = getattr(attn, norm_layer).register_forward_hook(
                                create_norm_hook(f'single_{i}', proj_type)
                            )
                            self.hooks.append(hook)
                    # V hook
                    is_fused = getattr(attn, 'fused_projections', False)
                    if is_fused and hasattr(attn, 'to_qkv'):
                        hook = attn.to_qkv.register_forward_hook(
                            create_v_hook(f'single_{i}', is_fused=True)
                        )
                    elif hasattr(attn, 'to_v'):
                        hook = attn.to_v.register_forward_hook(
                            create_v_hook(f'single_{i}', is_fused=False)
                        )
                    else:
                        continue
                    self.hooks.append(hook)

    def clear(self):
        """Clear collected data (keep hooks alive)."""
        self.qkv_data = {}

    def remove_hooks(self):
        """Remove all hooks and clear data."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.qkv_data = {}


# ═══════════════════════════════════════════════════════════════════════════════
# Compute context proportion + v_context velocity
# ═══════════════════════════════════════════════════════════════════════════════

def compute_context_metrics(qkv_data, image_rotary_emb, num_heads):
    """
    Compute context attention proportion and context velocity vectors.

    V from hook is raw [B, S, H*D], needs reshape to [B, S, H, D].
    Q, K from hook are already [B, S, H, D] (post norm_q/norm_k).

    Returns:
        context_prop: scalar — mean attention proportion on context
        velocities: dict {layer_key: [B, H, T_len, D]} — v_context per layer
    """
    proportions: List[torch.Tensor] = []
    velocities: Dict[str, torch.Tensor] = {}

    layer_keys = sorted({
        k.rsplit("_", 1)[0] for k in qkv_data if k.endswith("_q")
    })

    for lk in layer_keys:
        q = qkv_data.get(f"{lk}_q")
        k = qkv_data.get(f"{lk}_k")
        v_raw = qkv_data.get(f"{lk}_v")
        if q is None or k is None or v_raw is None:
            continue

        batch_size, seq_len, heads, dim_head = q.shape

        if 'single_' in lk and seq_len == 2560:
            tgt_s, tgt_e = 512, 1536
            ctx_s, ctx_e = 1536, 2560
        elif 'double_' in lk and seq_len == 2048:
            tgt_s, tgt_e = 0, 1024
            ctx_s, ctx_e = 1024, 2048
        else:
            continue

        # Reshape V: [B, S, H*D] → [B, S, H, D]
        v = v_raw.unflatten(-1, (num_heads, dim_head))

        # Apply RoPE to Q, K
        q_r = apply_rotary_emb(q, image_rotary_emb, sequence_dim=1)
        k_r = apply_rotary_emb(k, image_rotary_emb, sequence_dim=1)

        # Attention: target queries vs all keys
        q_tgt = q_r[:, tgt_s:tgt_e].permute(0, 2, 1, 3)   # [B, H, T, D]
        k_all = k_r.permute(0, 2, 1, 3)                     # [B, H, S, D]

        scores = q_tgt @ k_all.transpose(-2, -1) / math.sqrt(dim_head)
        weights = F.softmax(scores, dim=-1)

        # Context proportion
        w_ctx = weights[:, :, :, ctx_s:ctx_e]               # [B, H, T, C]
        proportions.append(w_ctx.sum(dim=-1).mean())

        # Context velocity: w_ctx @ V_ctx
        v_ctx = v[:, ctx_s:ctx_e].permute(0, 2, 1, 3)       # [B, H, C, D]
        velocities[lk] = w_ctx @ v_ctx                       # [B, H, T, D]

        del scores, weights, w_ctx

    if not proportions:
        device = image_rotary_emb[0].device
        dtype = image_rotary_emb[0].dtype
        return torch.tensor(0.0, device=device, dtype=dtype), {}

    return torch.stack(proportions).mean(), velocities


# ═══════════════════════════════════════════════════════════════════════════════
# Hijack target
# ═══════════════════════════════════════════════════════════════════════════════

def prepare_hijack_target(mode, resolution, device, dtype, path=None):
    if mode == "noise":
        return torch.randn(1, 3, resolution, resolution, device=device, dtype=dtype).clamp(0,1) * 2 - 1
    elif mode == "image" and path is not None:
        tf = transforms.Compose([
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        img = Image.open(path).convert("RGB")
        return tf(img).unsqueeze(0).to(device=device, dtype=dtype)
    else:
        return torch.randn(1, 3, resolution, resolution, device=device, dtype=dtype).clamp(0,1) * 2 - 1


# ═══════════════════════════════════════════════════════════════════════════════
# Prompt pool
# ═══════════════════════════════════════════════════════════════════════════════

def get_prompt_pool():
    return [
        "Make this person smile happily.",
        "Make this person look angry.",
        "Make this person look surprised.",
        "Make this person look sad.",
        "Give this person a serious expression.",
        "Make this person laugh joyfully.",
        "Make this person look worried.",
        "Make this person look excited.",
        "Make this person look tired.",
        "Make this person look confused.",
        "Add glasses to this person.",
        "Add sunglasses to this person.",
        "Add a hat to this person.",
        "Add a scarf to this person.",
        "Add a necklace to this person.",
        "Add earrings to this person.",
        "Add a cap to this person.",
        "Add a headband to this person.",
        "Add a watch to this person.",
        "Add a tie to this person.",
        "Make this person wave their hand.",
        "Make this person cross their arms.",
        "Make this person point forward.",
        "Make this person give a thumbs up.",
        "Make this person put hands on hips.",
        "Make this person cover their mouth.",
        "Make this person hold up a peace sign.",
        "Make this person shrug their shoulders.",
        "Make this person touch their chin.",
        "Make this person salute.",
        "Make this person look older.",
        "Make this person look younger.",
        "Add a beard to this person.",
        "Add a mustache to this person.",
        "Make this person's hair longer.",
        "Make this person's hair shorter.",
        "Add makeup to this person.",
        "Add freckles to this person.",
        "Make this person's skin tanned.",
        "Add stubble to this person.",
        "Place this person on a tropical beach with palm trees and ocean waves.",
        "Put this person in a snowy mountain landscape with pine trees.",
        "Move this person to a bustling city street with skyscrapers.",
        "Put this person in a beach setting.",
        "Put this person in a snowy mountain.",
        "Put this person in a city street.",
        "Put this person in a forest.",
        "Put this person in a coffee shop.",
        "Put this person in a library.",
        "Put this person in a park.",
        "Make this person smile and wave.",
        "Add glasses and make this person look serious.",
        "Make this person look older with a beard.",
        "Add a hat and make this person point forward.",
        "Make this person look surprised with hands on face.",
        "Add sunglasses and make this person give a thumbs up.",
        "Make this person look younger and smile happily.",
        "Add a scarf and make this person cross their arms.",
        "Make this person look tired and rub their eyes.",
        "Add earrings and make this person look confident.",
    ]


# ═══════════════════════════════════════════════════════════════════════════════
# Args
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model_name_or_path", type=str,
                        default="./models/FLUX.1-Kontext-dev")
    parser.add_argument("--condition_images_dir", type=str, default="./data/input")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--alpha", type=float, default=0.005)
    parser.add_argument("--eps", type=float, default=0.1)
    parser.add_argument("--attack_steps", type=int, default=200)
    parser.add_argument("--output_dir", type=str, default="./outputs/attack")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mixed_precision", type=str, default="bf16",
                        choices=["no", "fp16", "bf16"])
    parser.add_argument("--max_sequence_length", type=int, default=512)
    parser.add_argument("--hijack_mode", type=str, default="image",
                        choices=["noise", "image"])
    parser.add_argument("--hijack_image_path", type=str, default=str(Path(__file__).resolve().parents[2] / 'assets' / 'logo.png'))
    parser.add_argument("--lambda_attn", type=float, default=0.005) # 0.005
    parser.add_argument("--lambda_vel", type=float, default=1)
    parser.add_argument("--lambda_repel", type=float, default=1,
                        help="Weight for repelling adv v_ctx from clean v_ctx")
    parser.add_argument("--num_attack_layers", type=int, default=25,
                        help="Number of random layers to use per step")
    return parser.parse_args()


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    args = parse_args()

    accelerator = Accelerator(mixed_precision=args.mixed_precision, cpu=False)
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(message)s",
                        datefmt="%H:%M:%S", level=logging.INFO)

    if args.seed is not None:
        set_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    prompt_pool = get_prompt_pool()
    logger.info(f"Loaded {len(prompt_pool)} prompts")

    # ── Load models ───────────────────────────────────────────────────────────
    path = args.pretrained_model_name_or_path
    tokenizer_clip = CLIPTokenizer.from_pretrained(path, subfolder="tokenizer")
    tokenizer_t5 = T5TokenizerFast.from_pretrained(path, subfolder="tokenizer_2")
    text_encoder_clip = CLIPTextModel.from_pretrained(path, subfolder="text_encoder")
    text_encoder_t5 = T5EncoderModel.from_pretrained(path, subfolder="text_encoder_2")
    vae = AutoencoderKL.from_pretrained(path, subfolder="vae")
    transformer = FluxTransformer2DModel.from_pretrained(path, subfolder="transformer")
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(path, subfolder="scheduler")

    weight_dtype = {"no": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[args.mixed_precision]

    for m in (text_encoder_clip, text_encoder_t5, vae, transformer):
        m.to(accelerator.device, dtype=weight_dtype)
        m.eval()
        m.requires_grad_(False)

    transformer.enable_gradient_checkpointing()
    vae.enable_gradient_checkpointing()

    num_heads = transformer.config.num_attention_heads
    num_single_layers = transformer.config.num_single_layers  # 38

    guidance = torch.tensor([3.5], device=accelerator.device, dtype=weight_dtype)
    vae_shift = vae.config.shift_factor
    vae_scale = vae.config.scaling_factor

    # ── Pre-encode prompts ────────────────────────────────────────────────────
    text_encoders = [text_encoder_clip, text_encoder_t5]
    tokenizers = [tokenizer_clip, tokenizer_t5]

    prompt_embeds_pool = []
    pooled_embeds_pool = []
    for prompt in prompt_pool:
        with torch.no_grad():
            pe, ppe, _ = encode_prompt_flux_kontext(
                text_encoders, tokenizers, prompt, args.max_sequence_length, accelerator.device
            )
            prompt_embeds_pool.append(pe)
            pooled_embeds_pool.append(ppe)

    del text_encoder_clip, text_encoder_t5, tokenizer_clip, tokenizer_t5
    torch.cuda.empty_cache()

    # ── Prepare hook ──────────────────────────────────────────────────────────
    attention_hook = AttentionHook()

    # ── Dataset ───────────────────────────────────────────────────────────────
    dataset = AdversarialDataset(
        condition_images_dir=args.condition_images_dir,
        prompts=prompt_pool,
        resolution=args.resolution,
    )
    logger.info(f"Loaded {len(dataset)} condition images")

    # ── Prepare hijack target ─────────────────────────────────────────────────
    hijack_target = prepare_hijack_target(
        args.hijack_mode, args.resolution, accelerator.device, weight_dtype,
        path=args.hijack_image_path,
    )
    with torch.no_grad():
        hijack_lat = vae.encode(hijack_target).latent_dist.mode()
        hijack_lat = (hijack_lat - vae_shift) * vae_scale
    hijack_lat_ids = FluxKontextPipeline._prepare_latent_image_ids(
        hijack_lat.shape[0], hijack_lat.shape[2] // 2, hijack_lat.shape[3] // 2,
        accelerator.device, weight_dtype,
    )
    hijack_lat_ids[..., 0] = 1
    packed_hijack = FluxKontextPipeline._pack_latents(
        hijack_lat, 1, hijack_lat.shape[1], hijack_lat.shape[2], hijack_lat.shape[3],
    )

    # ── Attack loop ───────────────────────────────────────────────────────────
    for idx in range(len(dataset)):
        # if idx < 8:
        #     continue
        sample = dataset[idx]
        condition_image = sample['condition_image'].unsqueeze(0).to(accelerator.device, dtype=weight_dtype)
        cond_path = sample['cond_path']

        logger.info(f"Processing {idx+1}/{len(dataset)}: {Path(cond_path).name}")

        perturbed_condition = condition_image.clone().detach()
        original_condition = condition_image.clone().detach()

        # Fixed noise for this image
        fixed_noise = torch.randn(
            (1, 16, args.resolution // 8, args.resolution // 8),
            device=accelerator.device, dtype=weight_dtype,
        )

        progress_bar = tqdm(range(args.attack_steps), desc=f"Attacking {Path(cond_path).name}")

        for step in progress_bar:
            # Random prompt
            pi = torch.randint(0, len(prompt_pool), (1,)).item()
            prompt_embeds = prompt_embeds_pool[pi]
            pooled_embeds = pooled_embeds_pool[pi]
            text_ids = torch.zeros(prompt_embeds.shape[1], 3,
                                   device=accelerator.device, dtype=weight_dtype)

            # Random timestep
            timesteps = torch.randint(980, scheduler.config.num_train_timesteps,
                                      (1,), device=accelerator.device).long() # mid 500-520

            # timesteps = torch.randint(100, 120,(1,), device=accelerator.device).long()

            # Random layer subset
            active_indices = sorted(
                torch.randperm(num_single_layers)[:args.num_attack_layers].tolist()
            )

            # Prepare common latent inputs
            latent_image_ids = FluxKontextPipeline._prepare_latent_image_ids(
                1, fixed_noise.shape[2] // 2, fixed_noise.shape[3] // 2,
                accelerator.device, weight_dtype,
            )
            packed_noisy = FluxKontextPipeline._pack_latents(
                fixed_noise, 1, fixed_noise.shape[1], fixed_noise.shape[2], fixed_noise.shape[3],
            )
            timesteps_in = timesteps.expand(1).to(packed_noisy.dtype)

            # ── Step A: Hijack anchor forward (no grad) → ref velocities ──
            attention_hook.remove_hooks()
            attention_hook.register_hooks(transformer, {'single_blocks': active_indices})
            attention_hook.clear()

            packed_hijack_input = torch.cat([packed_noisy, packed_hijack], dim=1)
            combined_hijack_ids = torch.cat([latent_image_ids, hijack_lat_ids], dim=0)
            ids_hijack = torch.cat([text_ids, latent_image_ids, hijack_lat_ids], dim=0)
            rope_hijack = transformer.pos_embed(ids_hijack)

            with torch.no_grad():
                _ = transformer(
                    hidden_states=packed_hijack_input,
                    timestep=timesteps_in / 1000,
                    guidance=guidance.expand(1),
                    pooled_projections=pooled_embeds,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=combined_hijack_ids,
                    return_dict=False,
                )[0]

                _, ref_velocities = compute_context_metrics(
                    attention_hook.qkv_data, rope_hijack, num_heads,
                )
                ref_velocities = {k: v.detach() for k, v in ref_velocities.items()}

            # ── Step A': Clean condition forward (no grad) → clean velocities ─
            attention_hook.clear()

            with torch.no_grad():
                clean_lat = vae.encode(original_condition).latent_dist.mode()
                clean_lat = (clean_lat - vae_shift) * vae_scale

                clean_lat_ids = FluxKontextPipeline._prepare_latent_image_ids(
                    clean_lat.shape[0], clean_lat.shape[2] // 2, clean_lat.shape[3] // 2,
                    accelerator.device, weight_dtype,
                )
                clean_lat_ids[..., 0] = 1

                packed_clean = FluxKontextPipeline._pack_latents(
                    clean_lat, 1, clean_lat.shape[1], clean_lat.shape[2], clean_lat.shape[3],
                )
                packed_clean_input = torch.cat([packed_noisy, packed_clean], dim=1)
                combined_clean_ids = torch.cat([latent_image_ids, clean_lat_ids], dim=0)

                ids_clean = torch.cat([text_ids, latent_image_ids, clean_lat_ids], dim=0)
                rope_clean = transformer.pos_embed(ids_clean)

                _ = transformer(
                    hidden_states=packed_clean_input,
                    timestep=timesteps_in / 1000,
                    guidance=guidance.expand(1),
                    pooled_projections=pooled_embeds,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=combined_clean_ids,
                    return_dict=False,
                )[0]

                _, clean_velocities = compute_context_metrics(
                    attention_hook.qkv_data, rope_clean, num_heads,
                )
                clean_velocities = {k: v.detach() for k, v in clean_velocities.items()}

            # ── Step B: Perturbed condition forward (with grad) ───────────
            attention_hook.clear()
            perturbed_condition.requires_grad_(True)

            cond_lat = vae.encode(perturbed_condition).latent_dist.mode()
            cond_lat = (cond_lat - vae_shift) * vae_scale

            cond_lat_ids = FluxKontextPipeline._prepare_latent_image_ids(
                cond_lat.shape[0], cond_lat.shape[2] // 2, cond_lat.shape[3] // 2,
                accelerator.device, weight_dtype,
            )
            cond_lat_ids[..., 0] = 1

            packed_cond = FluxKontextPipeline._pack_latents(
                cond_lat, 1, cond_lat.shape[1], cond_lat.shape[2], cond_lat.shape[3],
            )
            packed_input = torch.cat([packed_noisy, packed_cond], dim=1)
            combined_ids = torch.cat([latent_image_ids, cond_lat_ids], dim=0)

            ids_pert = torch.cat([text_ids, latent_image_ids, cond_lat_ids], dim=0)
            rope_pert = transformer.pos_embed(ids_pert)

            _ = transformer(
                hidden_states=packed_input,
                timestep=timesteps_in / 1000,
                guidance=guidance.expand(1),
                pooled_projections=pooled_embeds,
                encoder_hidden_states=prompt_embeds,
                txt_ids=text_ids,
                img_ids=combined_ids,
                return_dict=False,
            )[0]

            context_prop, cur_velocities = compute_context_metrics(
                attention_hook.qkv_data, rope_pert, num_heads,
            )

            # ── Loss ──────────────────────────────────────────────────────
            loss_attn = 1.0 - context_prop

            # L_vel: pull adv v_ctx toward hijack ref
            loss_vel = torch.tensor(0.0, device=accelerator.device, dtype=weight_dtype)
            n_vel = 0
            for lk in cur_velocities:
                if lk in ref_velocities:
                    loss_vel = loss_vel + F.mse_loss(cur_velocities[lk], ref_velocities[lk])
                    n_vel += 1
            if n_vel > 0:
                loss_vel = loss_vel / n_vel

            # L_repel: push adv v_ctx away from clean v_ctx (negative MSE)
            loss_repel = torch.tensor(0.0, device=accelerator.device, dtype=weight_dtype)
            n_repel = 0
            for lk in cur_velocities:
                if lk in clean_velocities:
                    loss_repel = loss_repel + F.mse_loss(cur_velocities[lk], clean_velocities[lk])
                    n_repel += 1
            if n_repel > 0:
                loss_repel = loss_repel / n_repel
            # lambda_attn = random.uniform(0.001, 0.01)
            loss = (args.lambda_attn * loss_attn
                    + args.lambda_vel * loss_vel
                    - args.lambda_repel * loss_repel)

            # ── Backward + PGD ────────────────────────────────────────────
            transformer.zero_grad()
            vae.zero_grad()
            loss.backward()

            adv = perturbed_condition - args.alpha * perturbed_condition.grad.sign()
            eta = torch.clamp(adv - original_condition, -args.eps, args.eps)
            perturbed_condition = torch.clamp(original_condition + eta, -1, 1).detach_()

            pert_norm = (perturbed_condition - original_condition).abs().max().item()
            progress_bar.set_postfix(
                loss=f"{loss.item():.4f}",
                L_attn=f"{loss_attn.item():.4f}",
                L_vel=f"{loss_vel.item():.4f}",
                L_rep=f"{loss_repel.item():.4f}",
                ctx=f"{context_prop.item():.4f}",
                pert=f"{pert_norm:.4f}",
            )

            del (loss, loss_attn, loss_vel, loss_repel, cond_lat, packed_cond, packed_input,
                 adv, context_prop, cur_velocities, ref_velocities, clean_velocities,
                 packed_hijack_input)
            torch.cuda.empty_cache()

        # ── Save ──────────────────────────────────────────────────────────
        attention_hook.remove_hooks()

        with torch.no_grad():
            stem = Path(cond_path).stem
            out_path = os.path.join(args.output_dir, f"adversarial_{stem}.png")
            arr = perturbed_condition.squeeze(0).cpu().float().numpy()
            arr = ((arr + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
            arr = np.transpose(arr, (1, 2, 0))
            Image.fromarray(arr).save(out_path)
            logger.info(f"Saved: {out_path}")

        del perturbed_condition, original_condition, fixed_noise
        torch.cuda.empty_cache()

    logger.info("Attack completed!")


if __name__ == "__main__":
    main()
