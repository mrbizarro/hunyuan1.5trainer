#!/usr/bin/env python3
"""
Local HunyuanVideo 1.5 LoRA Training Script
Based on official musubi-tuner docs: https://github.com/kohya-ss/musubi-tuner/blob/main/docs/hunyuan_video_1_5.md

Run: python train_local.py --input_zip /path/to/data.zip --task t2v
"""
import os
import shutil
import subprocess
import zipfile
import argparse
import random
from pathlib import Path

# Paths - override with env vars
MODEL_CACHE = os.environ.get("MODEL_CACHE", "/src/models")
INPUT_DIR = os.environ.get("INPUT_DIR", "/src/input")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/src/output")
MUSUBI_DIR = os.environ.get("MUSUBI_DIR", "/src/musubi-tuner")

# Use ORIGINAL bf16 weights from tencent, NOT ComfyUI fp16
# Per docs: "do not use [ComfyUI weights] for bf16 training as the weights are converted to fp16"
MODEL_FILES = {
    # DiT - original bf16 from tencent
    "dit_t2v": "https://huggingface.co/tencent/HunyuanVideo-1.5/resolve/main/transformer/720p_t2v/diffusion_pytorch_model.safetensors",
    "dit_i2v": "https://huggingface.co/tencent/HunyuanVideo-1.5/resolve/main/transformer/720p_i2v/diffusion_pytorch_model.safetensors",
    # VAE - original from tencent
    "vae": "https://huggingface.co/tencent/HunyuanVideo-1.5/resolve/main/vae/diffusion_pytorch_model.safetensors",
    # Text encoders - these are fine from ComfyUI
    "text_encoder": "https://huggingface.co/Comfy-Org/HunyuanVideo_1.5_repackaged/resolve/main/split_files/text_encoders/qwen_2.5_vl_7b.safetensors",
    "byt5": "https://huggingface.co/Comfy-Org/HunyuanVideo_1.5_repackaged/resolve/main/split_files/text_encoders/byt5_small_glyphxl_fp16.safetensors",
    # Image encoder for I2V
    "image_encoder": "https://huggingface.co/Comfy-Org/HunyuanVideo_1.5_repackaged/resolve/main/split_files/clip_vision/sigclip_vision_patch14_384.safetensors",
}

# Filenames we'll save as (match what training script expects)
MODEL_FILENAMES = {
    "dit_t2v": "dit_t2v.safetensors",
    "dit_i2v": "dit_i2v.safetensors", 
    "vae": "vae.safetensors",
    "text_encoder": "qwen_2.5_vl_7b.safetensors",
    "byt5": "byt5_small_glyphxl_fp16.safetensors",
    "image_encoder": "sigclip_vision_patch14_384.safetensors",
}


def download_weights(task: str):
    """Download model weights if not cached"""
    print("\n" + "="*50)
    print("📥 Downloading Model Weights")
    print("="*50)
    os.makedirs(MODEL_CACHE, exist_ok=True)
    
    # Only download what we need
    needed = ["vae", "text_encoder", "byt5"]
    needed.append("dit_t2v" if task == "t2v" else "dit_i2v")
    if task == "i2v":
        needed.append("image_encoder")
    
    for name in needed:
        url = MODEL_FILES[name]
        filename = MODEL_FILENAMES[name]
        filepath = os.path.join(MODEL_CACHE, filename)
        
        if os.path.exists(filepath):
            size_gb = os.path.getsize(filepath) / (1024**3)
            print(f"✓ {name}: Already cached ({size_gb:.1f}GB)")
            continue
            
        print(f"⬇️ Downloading {name} from {url[:50]}...")
        subprocess.run(["wget", "--progress=bar:force", "-O", filepath, url], check=True)
        size_gb = os.path.getsize(filepath) / (1024**3)
        print(f"✓ {name}: Downloaded ({size_gb:.1f}GB)")
    
    print("✅ All weights ready\n")


def extract_zip(zip_path: str, extract_to: str):
    """Extract training data ZIP"""
    print("="*50)
    print("📂 Extracting Training Data")
    print("="*50)
    os.makedirs(extract_to, exist_ok=True)
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    
    # Count files (handle nested dirs)
    all_files = list(Path(extract_to).rglob("*"))
    video_exts = {'.mp4', '.mov', '.avi', '.webm', '.mkv'}
    image_exts = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
    
    videos = [f for f in all_files if f.suffix.lower() in video_exts]
    images = [f for f in all_files if f.suffix.lower() in image_exts]
    captions = [f for f in all_files if f.suffix.lower() == '.txt']
    
    print(f"📹 Videos: {len(videos)}")
    print(f"🖼️ Images: {len(images)}")  
    print(f"📝 Captions: {len(captions)}")
    
    # Move files from subdirs to root if needed
    for f in all_files:
        if f.is_file() and f.parent != Path(extract_to):
            dest = Path(extract_to) / f.name
            if not dest.exists():
                shutil.move(str(f), str(dest))
                print(f"  Moved: {f.name}")
    
    # Cleanup empty subdirs
    for d in sorted(Path(extract_to).rglob("*"), reverse=True):
        if d.is_dir() and not any(d.iterdir()):
            d.rmdir()
    
    # Determine dataset type
    is_video = len(videos) > 0
    print(f"\nDataset type: {'VIDEO' if is_video else 'IMAGE'}")
    print("✅ Extraction complete\n")
    
    return is_video


def create_dataset_toml(data_dir: str, is_video: bool, resolution: tuple = (544, 960)):
    """Create dataset configuration TOML per musubi-tuner format"""
    print("="*50)
    print("📝 Creating Dataset Config")
    print("="*50)
    
    cache_dir = os.path.join(data_dir, "cache")
    os.makedirs(cache_dir, exist_ok=True)
    
    # Per musubi-tuner docs - same structure for image/video
    if is_video:
        config = f"""[general]
resolution = [{resolution[0]}, {resolution[1]}]
caption_extension = ".txt"
batch_size = 1
enable_bucket = true
bucket_no_upscale = false

[[datasets]]
video_directory = "{data_dir}"
cache_directory = "{cache_dir}"
target_frames = [1, 13, 25]
frame_extraction = "head"
"""
    else:
        config = f"""[general]
resolution = [{resolution[0]}, {resolution[1]}]
caption_extension = ".txt"
batch_size = 1
enable_bucket = true
bucket_no_upscale = false

[[datasets]]
image_directory = "{data_dir}"
cache_directory = "{cache_dir}"
"""
    
    toml_path = os.path.join(OUTPUT_DIR, "train.toml")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(toml_path, "w") as f:
        f.write(config)
    
    print(f"Saved: {toml_path}")
    print(f"Contents:\n{config}")
    print("="*50 + "\n")
    return toml_path


def cache_latents(toml_path: str, task: str):
    """Cache VAE latents using hv_1_5_cache_latents.py"""
    print("="*50)
    print("💾 Caching Latents (VAE)")
    print("="*50)
    
    vae_path = os.path.join(MODEL_CACHE, MODEL_FILENAMES["vae"])
    
    cmd = [
        "python", f"{MUSUBI_DIR}/hv_1_5_cache_latents.py",
        "--dataset_config", toml_path,
        "--vae", vae_path,
        "--vae_sample_size", "128",  # Use tiling for memory, 256 for quality if VRAM allows
    ]
    
    # For I2V, cache image features too
    if task == "i2v":
        image_encoder_path = os.path.join(MODEL_CACHE, MODEL_FILENAMES["image_encoder"])
        cmd.extend(["--i2v", "--image_encoder", image_encoder_path])
    
    print(f"Command: {' '.join(cmd)}\n")
    result = subprocess.run(cmd, cwd=MUSUBI_DIR)
    if result.returncode != 0:
        raise RuntimeError(f"Latent caching failed with code {result.returncode}")
    print("✅ Latents cached\n")


def cache_text_encoder_outputs(toml_path: str):
    """Cache text encoder outputs using hv_1_5_cache_text_encoder_outputs.py"""
    print("="*50)
    print("💭 Caching Text Encoder Outputs")
    print("="*50)
    
    text_encoder_path = os.path.join(MODEL_CACHE, MODEL_FILENAMES["text_encoder"])
    byt5_path = os.path.join(MODEL_CACHE, MODEL_FILENAMES["byt5"])
    
    cmd = [
        "python", f"{MUSUBI_DIR}/hv_1_5_cache_text_encoder_outputs.py",
        "--dataset_config", toml_path,
        "--text_encoder", text_encoder_path,
        "--byt5", byt5_path,
        "--batch_size", "4",
        "--fp8_vl",  # Use FP8 for VRAM savings per docs
    ]
    
    print(f"Command: {' '.join(cmd)}\n")
    result = subprocess.run(cmd, cwd=MUSUBI_DIR)
    if result.returncode != 0:
        raise RuntimeError(f"Text encoder caching failed with code {result.returncode}")
    print("✅ Text encodings cached\n")


def run_training(args, toml_path: str):
    """Run LoRA training using hv_1_5_train_network.py
    
    Command based on official musubi-tuner docs:
    https://github.com/kohya-ss/musubi-tuner/blob/main/docs/hunyuan_video_1_5.md
    """
    print("="*50)
    print("🚀 Starting LoRA Training")
    print("="*50)
    
    dit_file = MODEL_FILENAMES["dit_t2v"] if args.task == "t2v" else MODEL_FILENAMES["dit_i2v"]
    dit_path = os.path.join(MODEL_CACHE, dit_file)
    vae_path = os.path.join(MODEL_CACHE, MODEL_FILENAMES["vae"])
    text_encoder_path = os.path.join(MODEL_CACHE, MODEL_FILENAMES["text_encoder"])
    byt5_path = os.path.join(MODEL_CACHE, MODEL_FILENAMES["byt5"])
    
    # Build command per official docs
    cmd = [
        "accelerate", "launch",
        "--num_cpu_threads_per_process", "1",  # docs use 1
        "--mixed_precision", "bf16",
        f"{MUSUBI_DIR}/hv_1_5_train_network.py",
        # Model paths
        "--dit", dit_path,
        "--vae", vae_path,
        "--text_encoder", text_encoder_path,
        "--byt5", byt5_path,
        "--dataset_config", toml_path,
        # Task
        "--task", args.task,
        # Attention & precision (per docs)
        "--sdpa",
        "--mixed_precision", "bf16",
        # Timestep sampling (per docs: "base on these and adjust")
        "--timestep_sampling", "shift",
        "--weighting_scheme", "none",
        "--discrete_flow_shift", "2.0",
        # Optimizer
        "--optimizer_type", args.optimizer,
        "--learning_rate", str(args.learning_rate),
        "--gradient_checkpointing",
        # Data loading
        "--max_data_loader_n_workers", "2",
        "--persistent_data_loader_workers",
        # LoRA config
        "--network_module", "networks.lora_hv_1_5",
        "--network_dim", str(args.rank),
        # Output
        "--seed", str(args.seed),
        "--output_dir", OUTPUT_DIR,
        "--output_name", "lora",
    ]
    
    # Memory optimization for 24GB cards
    if args.fp8:
        cmd.extend(["--fp8_base", "--fp8_scaled", "--fp8_vl"])
    
    # I2V needs image encoder
    if args.task == "i2v":
        image_encoder_path = os.path.join(MODEL_CACHE, MODEL_FILENAMES["image_encoder"])
        cmd.extend(["--image_encoder", image_encoder_path])
    
    # Epochs or steps
    if args.max_train_steps > 0:
        cmd.extend(["--max_train_steps", str(args.max_train_steps)])
    else:
        cmd.extend(["--max_train_epochs", str(args.epochs)])
    
    # Save checkpoints
    cmd.extend(["--save_every_n_epochs", "1"])
    
    print(f"Command:\n{' '.join(cmd)}\n")
    result = subprocess.run(cmd, cwd=MUSUBI_DIR)
    if result.returncode != 0:
        raise RuntimeError(f"Training failed with code {result.returncode}")
    print("\n✨ Training Complete!\n")


def add_trigger_word(data_dir: str, trigger_word: str):
    """Prepend trigger word to all caption files"""
    if not trigger_word:
        return
    print(f"Adding trigger word '{trigger_word}' to captions...")
    count = 0
    for txt_file in Path(data_dir).glob("*.txt"):
        content = txt_file.read_text().strip()
        txt_file.write_text(f"{trigger_word}, {content}")
        count += 1
    print(f"Updated {count} caption files\n")


def convert_to_comfyui():
    """Convert LoRA to ComfyUI format"""
    print("="*50)
    print("🔄 Converting LoRA to ComfyUI Format")
    print("="*50)
    
    input_lora = os.path.join(OUTPUT_DIR, "lora.safetensors")
    output_lora = os.path.join(OUTPUT_DIR, "lora_comfyui.safetensors")
    
    if not os.path.exists(input_lora):
        print(f"⚠️ LoRA not found at {input_lora}, skipping conversion")
        return
    
    cmd = [
        "python",
        f"{MUSUBI_DIR}/src/musubi_tuner/networks/convert_hunyuan_video_1_5_lora_to_comfy.py",
        input_lora,
        output_lora,
    ]
    
    print(f"Command: {' '.join(cmd)}\n")
    result = subprocess.run(cmd, cwd=MUSUBI_DIR)
    if result.returncode == 0:
        print(f"✅ ComfyUI LoRA saved to: {output_lora}\n")
    else:
        print(f"⚠️ Conversion failed (code {result.returncode})\n")


def main():
    parser = argparse.ArgumentParser(
        description="HunyuanVideo 1.5 LoRA Training (Local)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic training
  python train_local.py --input_zip /path/to/data.zip --task t2v --trigger_word "mystyle"
  
  # Quick test (1 epoch)
  python train_local.py --input_zip /path/to/data.zip --epochs 1 --skip_download
  
  # Resume with cached data
  python train_local.py --input_zip /path/to/data.zip --skip_download --skip_cache
"""
    )
    parser.add_argument("--input_zip", type=str, required=True, help="Path to training data ZIP")
    parser.add_argument("--task", type=str, default="t2v", choices=["t2v", "i2v"], help="t2v or i2v")
    parser.add_argument("--trigger_word", type=str, default="", help="Trigger word to prepend")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--max_train_steps", type=int, default=-1, help="Max steps (overrides epochs)")
    parser.add_argument("--rank", type=int, default=32, help="LoRA rank (4-128)")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--optimizer", type=str, default="adamw8bit", help="Optimizer")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (-1 for random)")
    parser.add_argument("--fp8", action="store_true", help="Use FP8 for memory savings (24GB cards)")
    parser.add_argument("--skip_download", action="store_true", help="Skip weight download")
    parser.add_argument("--skip_cache", action="store_true", help="Skip latent/text caching")
    parser.add_argument("--skip_convert", action="store_true", help="Skip ComfyUI conversion")
    
    args = parser.parse_args()
    
    if args.seed < 0:
        args.seed = random.randint(0, 2**16)
    
    print("\n" + "="*50)
    print("🎬 HunyuanVideo 1.5 LoRA Training")
    print("="*50)
    print(f"Input:    {args.input_zip}")
    print(f"Task:     {args.task}")
    print(f"Trigger:  {args.trigger_word or '(none)'}")
    print(f"Epochs:   {args.epochs}")
    print(f"Rank:     {args.rank}")
    print(f"LR:       {args.learning_rate}")
    print(f"Optimizer:{args.optimizer}")
    print(f"FP8:      {args.fp8}")
    print(f"Seed:     {args.seed}")
    print("="*50 + "\n")
    
    # Setup dirs
    data_dir = os.path.join(INPUT_DIR, "data")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 1. Download weights
    if not args.skip_download:
        download_weights(args.task)
    
    # 2. Extract data
    is_video = extract_zip(args.input_zip, data_dir)
    
    # 3. Add trigger word
    add_trigger_word(data_dir, args.trigger_word)
    
    # 4. Create config
    toml_path = create_dataset_toml(data_dir, is_video)
    
    if not args.skip_cache:
        # 5. Cache latents
        cache_latents(toml_path, args.task)
        
        # 6. Cache text encodings
        cache_text_encoder_outputs(toml_path)
    
    # 7. Train
    run_training(args, toml_path)
    
    # 8. Convert to ComfyUI format
    if not args.skip_convert:
        convert_to_comfyui()
    
    print("\n" + "="*50)
    print("✅ All Done!")
    print(f"   LoRA: {OUTPUT_DIR}/lora.safetensors")
    print(f"   ComfyUI: {OUTPUT_DIR}/lora_comfyui.safetensors")
    print("="*50 + "\n")


if __name__ == "__main__":
    main()
