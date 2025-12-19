#!/usr/bin/env python3
"""
Local HunyuanVideo 1.5 LoRA Training Script
Usage: python train_local.py --data ./my_videos --trigger cinematron --steps 1000
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

# Add musubi-tuner to path
MUSUBI_DIR = Path(__file__).parent / "musubi-tuner"
sys.path.insert(0, str(MUSUBI_DIR))

WEIGHTS_DIR = Path(__file__).parent / "weights"

def find_script(names):
    """Find a musubi-tuner script."""
    for name in names:
        for loc in [MUSUBI_DIR / name, MUSUBI_DIR / "src" / "musubi_tuner" / name]:
            if loc.exists():
                return loc
    return None

def create_dataset_config(data_dir, output_path, w, h):
    """Create TOML config for dataset."""
    Path(output_path).write_text(f'''[general]
shuffle_caption = true
caption_extension = ".txt"

[[datasets]]
batch_size = 1
resolution = [{w}, {h}]
enable_bucket = true

  [[datasets.subsets]]
  image_directory = "{data_dir}"
  cache_directory = "{data_dir}/cache"
  target_frames = [1, 25, 49]
''')

def main():
    parser = argparse.ArgumentParser(description="Train HunyuanVideo 1.5 LoRA locally")
    parser.add_argument("--data", required=True, help="Directory with videos/images + .txt captions")
    parser.add_argument("--trigger", default="TOK", help="Trigger word")
    parser.add_argument("--steps", type=int, default=1000, help="Training steps")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--rank", type=int, default=32, help="LoRA rank")
    parser.add_argument("--width", type=int, default=544, help="Resolution width")
    parser.add_argument("--height", type=int, default=960, help="Resolution height")
    parser.add_argument("--blocks-to-swap", type=int, default=20, help="CPU offload blocks (more=less VRAM)")
    parser.add_argument("--output", default="./output", help="Output directory")
    args = parser.parse_args()

    data_dir = Path(args.data).resolve()
    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Paths
    DIT_PATH = WEIGHTS_DIR / "hunyuan1.5_dit.safetensors"
    VAE_PATH = WEIGHTS_DIR / "hunyuan1.5_vae.safetensors"
    TE1_PATH = WEIGHTS_DIR / "text_encoder" / "qwen_2.5_vl_7b.safetensors"
    TE2_PATH = WEIGHTS_DIR / "text_encoder_2" / "byt5.safetensors"

    # Check weights exist
    for p in [DIT_PATH, VAE_PATH, TE1_PATH, TE2_PATH]:
        if not p.exists():
            print(f"ERROR: Missing weight file: {p}")
            print("Run: bash setup.sh")
            sys.exit(1)

    # Find scripts
    cache_latent = find_script(["hv15_cache_latents.py", "cache_latents.py"])
    cache_te = find_script(["hv15_cache_text_encoder_outputs.py", "cache_text_encoder_outputs.py"])
    train_script = find_script(["hv15_train_network.py", "hv_train_network.py"])

    if not all([cache_latent, cache_te, train_script]):
        print("ERROR: Missing musubi-tuner scripts. Run: bash setup.sh")
        sys.exit(1)

    print("=" * 60)
    print("HunyuanVideo 1.5 LoRA Training (Local)")
    print("=" * 60)
    print(f"Data: {data_dir}")
    print(f"Resolution: {args.width}x{args.height}")
    print(f"Steps: {args.steps}, LR: {args.lr}, Rank: {args.rank}")
    print(f"Blocks to swap: {args.blocks_to_swap}")
    print()

    # Create dataset config
    dataset_config = data_dir / "dataset.toml"
    create_dataset_config(str(data_dir), str(dataset_config), args.width, args.height)

    # Step 1: Cache latents
    print("[1/3] Caching latents...")
    subprocess.run([
        sys.executable, str(cache_latent),
        "--dataset_config", str(dataset_config),
        "--vae", str(VAE_PATH),
        "--vae_chunk_size", "32",
        "--vae_tiling",
    ], check=True, cwd=str(MUSUBI_DIR))

    # Step 2: Cache text encoder outputs
    print("\n[2/3] Caching text encoder outputs...")
    te_cmd = [
        sys.executable, str(cache_te),
        "--dataset_config", str(dataset_config),
        "--text_encoder1", str(TE1_PATH),
        "--text_encoder2", str(TE2_PATH),
        "--batch_size", "1",
    ]
    subprocess.run(te_cmd, check=True, cwd=str(MUSUBI_DIR))

    # Step 3: Train
    print("\n[3/3] Training LoRA...")
    train_cmd = [
        "accelerate", "launch",
        "--num_cpu_threads_per_process", "1",
        "--mixed_precision", "bf16",
        str(train_script),
        "--dit", str(DIT_PATH),
        "--dataset_config", str(dataset_config),
        "--sdpa",
        "--mixed_precision", "bf16",
        "--fp8_base",
        "--optimizer_type", "adamw8bit",
        "--learning_rate", str(args.lr),
        "--gradient_checkpointing",
        "--max_data_loader_n_workers", "2",
        "--network_module", "networks.lora",
        "--network_dim", str(args.rank),
        "--timestep_sampling", "shift",
        "--discrete_flow_shift", "7.0",
        "--max_train_steps", str(args.steps),
        "--save_every_n_steps", str(max(args.steps // 4, 100)),
        "--seed", "42",
        "--output_dir", str(output_dir),
        "--output_name", f"hunyuan15_{args.trigger}",
        "--blocks_to_swap", str(args.blocks_to_swap),
    ]
    subprocess.run(train_cmd, check=True, cwd=str(MUSUBI_DIR))

    print("\n" + "=" * 60)
    print("Training Complete!")
    print(f"Output: {output_dir}")
    print("=" * 60)

if __name__ == "__main__":
    main()
