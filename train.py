"""
HunyuanVideo 1.5 LoRA Training for Replicate
Based on kohya-ss/musubi-tuner
"""
import os
import shutil
import subprocess
import zipfile
import tarfile
import tempfile
import random
from pathlib import Path
from typing import Optional
from cog import BaseModel, Input, Path as CogPath

# Paths
MODEL_CACHE = "/src/models"
INPUT_DIR = "/src/input"
OUTPUT_DIR = "/src/output"
MUSUBI_DIR = "/src/musubi-tuner"

# Model URLs from HuggingFace/ComfyUI
MODEL_FILES = {
    "dit_t2v": "https://huggingface.co/Comfy-Org/HunyuanVideo_1.5_repackaged/resolve/main/split_files/diffusion_models/hunyuanvideo1.5_720p_t2v_fp16.safetensors",
    "dit_i2v": "https://huggingface.co/Comfy-Org/HunyuanVideo_1.5_repackaged/resolve/main/split_files/diffusion_models/hunyuanvideo1.5_720p_i2v_fp16.safetensors", 
    "vae": "https://huggingface.co/Comfy-Org/HunyuanVideo_1.5_repackaged/resolve/main/split_files/vae/hunyuanvideo15_vae_fp16.safetensors",
    "text_encoder": "https://huggingface.co/Comfy-Org/HunyuanVideo_1.5_repackaged/resolve/main/split_files/text_encoders/qwen_2.5_vl_7b.safetensors",
    "byt5": "https://huggingface.co/Comfy-Org/HunyuanVideo_1.5_repackaged/resolve/main/split_files/text_encoders/byt5_small_glyphxl_fp16.safetensors",
    "image_encoder": "https://huggingface.co/Comfy-Org/HunyuanVideo_1.5_repackaged/resolve/main/split_files/clip_vision/sigclip_vision_patch14_384.safetensors",
}

class TrainingOutput(BaseModel):
    weights: CogPath


def download_weights():
    """Download model weights if not cached"""
    print("\n=== 📥 Downloading Model Weights ===")
    os.makedirs(MODEL_CACHE, exist_ok=True)
    
    for name, url in MODEL_FILES.items():
        filename = url.split("/")[-1]
        filepath = os.path.join(MODEL_CACHE, filename)
        
        if os.path.exists(filepath):
            print(f"✓ {name}: Already cached")
            continue
            
        print(f"⬇️ Downloading {name}...")
        subprocess.run([
            "wget", "-q", "--show-progress", "-O", filepath, url
        ], check=True)
        print(f"✓ {name}: Downloaded")
    
    print("✅ All weights ready")
    print("=====================================")


def extract_zip(zip_path: str, extract_to: str):
    """Extract training data ZIP"""
    print("\n=== 📂 Extracting Training Data ===")
    os.makedirs(extract_to, exist_ok=True)
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    
    # Count files
    video_count = len([f for f in os.listdir(extract_to) if f.endswith(('.mp4', '.mov', '.avi', '.webm'))])
    image_count = len([f for f in os.listdir(extract_to) if f.endswith(('.jpg', '.jpeg', '.png', '.webp'))])
    caption_count = len([f for f in os.listdir(extract_to) if f.endswith('.txt')])
    
    print(f"📹 Videos: {video_count}")
    print(f"🖼️ Images: {image_count}")
    print(f"📝 Captions: {caption_count}")
    print("✅ Extraction complete")
    print("=====================================")
    
    return video_count > 0  # Returns True if video dataset


def create_dataset_toml(is_video: bool, resolution: tuple = (960, 544)):
    """Create dataset configuration TOML"""
    print("\n=== 📝 Creating Dataset Config ===")
    
    config = f"""[general]
resolution = [{resolution[0]}, {resolution[1]}]
caption_extension = ".txt"
batch_size = 1
enable_bucket = true
bucket_no_upscale = false

[[datasets]]
{"video_directory" if is_video else "image_directory"} = "{INPUT_DIR}/videos"
cache_directory = "{INPUT_DIR}/cache"
"""
    
    if is_video:
        config += """target_frames = [1, 13, 25]
frame_extraction = "head"
"""
    
    toml_path = "/src/train.toml"
    with open(toml_path, "w") as f:
        f.write(config)
    
    print(f"Config saved to {toml_path}")
    print("=====================================")
    return toml_path


def cache_latents(task: str):
    """Cache VAE latents"""
    print("\n=== 💾 Caching Latents ===")
    
    vae_path = os.path.join(MODEL_CACHE, "hunyuanvideo15_vae_fp16.safetensors")
    
    cmd = [
        "python", f"{MUSUBI_DIR}/hv_1_5_cache_latents.py",
        "--dataset_config", "/src/train.toml",
        "--vae", vae_path,
        "--vae_sample_size", "128",
    ]
    
    # For I2V, add image encoder
    if task == "i2v":
        image_encoder_path = os.path.join(MODEL_CACHE, "sigclip_vision_patch14_384.safetensors")
        cmd.extend(["--i2v", "--image_encoder", image_encoder_path])
    
    subprocess.run(cmd, check=True, cwd=MUSUBI_DIR)
    print("✅ Latents cached")
    print("=====================================")


def cache_text_encoder_outputs():
    """Cache text encoder outputs"""
    print("\n=== 💭 Caching Text Encodings ===")
    
    text_encoder_path = os.path.join(MODEL_CACHE, "qwen_2.5_vl_7b.safetensors")
    byt5_path = os.path.join(MODEL_CACHE, "byt5_small_glyphxl_fp16.safetensors")
    
    cmd = [
        "python", f"{MUSUBI_DIR}/hv_1_5_cache_text_encoder_outputs.py",
        "--dataset_config", "/src/train.toml",
        "--text_encoder", text_encoder_path,
        "--byt5", byt5_path,
        "--batch_size", "4",
        "--fp8_vl",  # Use FP8 to save VRAM
    ]
    
    subprocess.run(cmd, check=True, cwd=MUSUBI_DIR)
    print("✅ Text encodings cached")
    print("=====================================")


def run_training(
    task: str,
    epochs: int,
    max_train_steps: int,
    rank: int,
    learning_rate: float,
    optimizer: str,
    seed: int,
):
    """Run LoRA training"""
    print("\n=== 🚀 Starting Training ===")
    
    # Select DiT based on task
    if task == "t2v":
        dit_path = os.path.join(MODEL_CACHE, "hunyuanvideo1.5_720p_t2v_fp16.safetensors")
    else:
        dit_path = os.path.join(MODEL_CACHE, "hunyuanvideo1.5_720p_i2v_fp16.safetensors")
    
    vae_path = os.path.join(MODEL_CACHE, "hunyuanvideo15_vae_fp16.safetensors")
    text_encoder_path = os.path.join(MODEL_CACHE, "qwen_2.5_vl_7b.safetensors")
    byt5_path = os.path.join(MODEL_CACHE, "byt5_small_glyphxl_fp16.safetensors")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    cmd = [
        "accelerate", "launch",
        "--num_cpu_threads_per_process", "8",
        "--mixed_precision", "bf16",
        f"{MUSUBI_DIR}/hv_1_5_train_network.py",
        "--dit", dit_path,
        "--vae", vae_path,
        "--text_encoder", text_encoder_path,
        "--byt5", byt5_path,
        "--dataset_config", "/src/train.toml",
        "--task", task,
        "--sdpa",
        "--mixed_precision", "bf16",
        "--fp8_base", "--fp8_scaled",  # Memory optimization
        "--fp8_vl",  # FP8 for text encoder
        "--optimizer_type", optimizer,
        "--learning_rate", str(learning_rate),
        "--gradient_checkpointing",
        "--max_data_loader_n_workers", "4",
        "--persistent_data_loader_workers",
        "--network_module", "networks.lora_hv_1_5",
        "--network_dim", str(rank),
        "--timestep_sampling", "shift",
        "--discrete_flow_shift", "2.0",
        "--seed", str(seed),
        "--output_dir", OUTPUT_DIR,
        "--output_name", "lora",
    ]
    
    # Add image encoder for I2V
    if task == "i2v":
        image_encoder_path = os.path.join(MODEL_CACHE, "sigclip_vision_patch14_384.safetensors")
        cmd.extend(["--image_encoder", image_encoder_path])
    
    # Epochs or steps
    if max_train_steps > 0:
        cmd.extend(["--max_train_steps", str(max_train_steps)])
    else:
        cmd.extend(["--max_train_epochs", str(epochs)])
    
    subprocess.run(cmd, check=True, cwd=MUSUBI_DIR)
    print("\n✨ Training Complete!")
    print("=====================================")


def convert_lora_to_comfyui():
    """Convert LoRA to ComfyUI format"""
    print("\n=== 🔄 Converting LoRA Format ===")
    
    input_lora = os.path.join(OUTPUT_DIR, "lora.safetensors")
    output_lora = os.path.join(OUTPUT_DIR, "lora_comfyui.safetensors")
    
    cmd = [
        "python", 
        f"{MUSUBI_DIR}/src/musubi_tuner/networks/convert_hunyuan_video_1_5_lora_to_comfy.py",
        input_lora,
        output_lora,
    ]
    
    subprocess.run(cmd, check=True, cwd=MUSUBI_DIR)
    print("✅ LoRA converted to ComfyUI format")
    print("=====================================")


def archive_results() -> str:
    """Create output archive"""
    print("\n=== 📦 Archiving Results ===")
    
    archive_path = "/src/trained_lora.tar"
    
    with tarfile.open(archive_path, "w") as tar:
        # Add the original LoRA
        lora_path = os.path.join(OUTPUT_DIR, "lora.safetensors")
        if os.path.exists(lora_path):
            tar.add(lora_path, arcname="lora.safetensors")
        
        # Add ComfyUI format
        comfy_path = os.path.join(OUTPUT_DIR, "lora_comfyui.safetensors")
        if os.path.exists(comfy_path):
            tar.add(comfy_path, arcname="lora_comfyui.safetensors")
    
    print(f"✅ Archive created: {archive_path}")
    print("=====================================")
    return archive_path


def train(
    input_videos: CogPath = Input(
        description="ZIP file containing training videos/images with captions (.txt files with same name)"
    ),
    task: str = Input(
        description="Training task: t2v (text-to-video) or i2v (image-to-video)",
        default="t2v",
        choices=["t2v", "i2v"],
    ),
    trigger_word: str = Input(
        description="Trigger word for the LoRA (will be prepended to captions)",
        default="",
    ),
    epochs: int = Input(
        description="Number of training epochs",
        default=10,
        ge=1,
        le=100,
    ),
    max_train_steps: int = Input(
        description="Max training steps (overrides epochs if > 0)",
        default=-1,
        ge=-1,
        le=10000,
    ),
    rank: int = Input(
        description="LoRA rank (dimension)",
        default=32,
        ge=4,
        le=128,
    ),
    learning_rate: float = Input(
        description="Learning rate",
        default=1e-4,
        ge=1e-6,
        le=1e-3,
    ),
    optimizer: str = Input(
        description="Optimizer type",
        default="adamw8bit",
        choices=["adamw8bit", "adamw", "AdaFactor"],
    ),
    seed: int = Input(
        description="Random seed (-1 for random)",
        default=42,
        ge=-1,
    ),
) -> TrainingOutput:
    """Train HunyuanVideo 1.5 LoRA"""
    
    print("=" * 50)
    print("🎬 HunyuanVideo 1.5 LoRA Training")
    print("=" * 50)
    
    # Handle seed
    if seed < 0:
        seed = random.randint(0, 2**16)
    print(f"Using seed: {seed}")
    
    # Setup directories
    os.makedirs(INPUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(INPUT_DIR, "videos"), exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Download weights
    download_weights()
    
    # Extract training data
    is_video = extract_zip(str(input_videos), os.path.join(INPUT_DIR, "videos"))
    
    # Add trigger word to captions if specified
    if trigger_word:
        print(f"\n=== Adding trigger word: {trigger_word} ===")
        video_dir = os.path.join(INPUT_DIR, "videos")
        for txt_file in Path(video_dir).glob("*.txt"):
            content = txt_file.read_text()
            txt_file.write_text(f"{trigger_word}, {content}")
    
    # Create dataset config
    create_dataset_toml(is_video)
    
    # Cache latents
    cache_latents(task)
    
    # Cache text encoder outputs
    cache_text_encoder_outputs()
    
    # Run training
    run_training(
        task=task,
        epochs=epochs,
        max_train_steps=max_train_steps,
        rank=rank,
        learning_rate=learning_rate,
        optimizer=optimizer,
        seed=seed,
    )
    
    # Convert to ComfyUI format
    convert_lora_to_comfyui()
    
    # Archive results
    output_path = archive_results()
    
    print("\n" + "=" * 50)
    print("✅ Training Complete!")
    print("=" * 50)
    
    return TrainingOutput(weights=CogPath(output_path))
