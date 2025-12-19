"""
HunyuanVideo 1.5 LoRA Trainer for Replicate (A100 80GB)
Uses musubi-tuner for HunyuanVideo 1.5 training with Qwen2.5-VL + ByT5 text encoders
"""

import os
import sys
import subprocess
import zipfile
from pathlib import Path
from cog import BasePredictor, BaseModel, Input, Path as CogPath

sys.path.insert(0, "/src/musubi-tuner")

class TrainingOutput(BaseModel):
    weights: CogPath

class Predictor(BasePredictor):
    def setup(self):
        pass
    
    def predict(
        self,
        info: str = Input(description="This is a TRAINING model. Click the Train tab!", default="Click Train tab")
    ) -> str:
        return "Training-only model. Use the Train tab to fine-tune HunyuanVideo 1.5."

def setup_florence2():
    from transformers import AutoProcessor, AutoModelForCausalLM
    import torch
    processor = AutoProcessor.from_pretrained("microsoft/Florence-2-large", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained("microsoft/Florence-2-large", torch_dtype=torch.float16, trust_remote_code=True).cuda()
    return processor, model

def caption_image(image_path, processor, model, trigger_word):
    from PIL import Image
    import torch
    image = Image.open(image_path).convert("RGB")
    inputs = processor(text="<MORE_DETAILED_CAPTION>", images=image, return_tensors="pt").to("cuda", torch.float16)
    generated_ids = model.generate(input_ids=inputs["input_ids"], pixel_values=inputs["pixel_values"], max_new_tokens=256, num_beams=3)
    caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].replace("<MORE_DETAILED_CAPTION>", "").strip()
    return f"{trigger_word}, {caption}"

def extract_and_caption(zip_path, output_dir, trigger_word):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(output_dir)
    
    image_exts = {'.jpg', '.jpeg', '.png', '.webp'}
    video_exts = {'.mp4', '.mov', '.webm'}
    media_files = [f for f in output_dir.rglob("*") if f.suffix.lower() in image_exts | video_exts]
    
    needs_caption = [m for m in media_files if not m.with_suffix('.txt').exists()]
    
    if needs_caption:
        print(f"Auto-captioning {len(needs_caption)} files with Florence-2...")
        processor, model = setup_florence2()
        for media in needs_caption:
            if media.suffix.lower() in image_exts:
                caption = caption_image(media, processor, model, trigger_word)
            else:
                import cv2
                cap = cv2.VideoCapture(str(media))
                ret, frame = cap.read()
                cap.release()
                if ret:
                    temp_img = output_dir / "temp.jpg"
                    cv2.imwrite(str(temp_img), frame)
                    caption = caption_image(temp_img, processor, model, trigger_word)
                    temp_img.unlink()
                else:
                    caption = f"{trigger_word}, a video"
            media.with_suffix('.txt').write_text(caption)
            print(f"  {media.name}: {caption[:60]}...")
        del processor, model
        import torch; torch.cuda.empty_cache()
    else:
        print("All files have .txt captions - using your captions")
    return output_dir, media_files

def create_dataset_config(data_dir, output_path, w, h):
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

def find_script(musubi_dir, script_names):
    """Find a script in musubi-tuner directory, checking multiple possible locations."""
    for name in script_names:
        # Check root
        path = musubi_dir / name
        if path.exists():
            return path
        # Check src/musubi_tuner
        path = musubi_dir / "src" / "musubi_tuner" / name
        if path.exists():
            return path
    return None

def train(
    input_videos: CogPath = Input(description="ZIP file with videos/images + optional .txt captions"),
    trigger_word: str = Input(description="Trigger word for LoRA", default="TOK"),
    train_steps: int = Input(description="Training steps", default=1000, ge=100, le=5000),
    learning_rate: float = Input(description="Learning rate", default=2e-4),
    lora_rank: int = Input(description="LoRA rank", default=32, ge=8, le=128),
    resolution_width: int = Input(description="Video width", default=544, ge=256, le=1280),
    resolution_height: int = Input(description="Video height", default=960, ge=256, le=1280),
    blocks_to_swap: int = Input(description="CPU offload blocks (more=less VRAM)", default=32, ge=0, le=40),
) -> TrainingOutput:
    
    # Paths for HunyuanVideo 1.5
    WEIGHTS_DIR = Path("/src/weights")
    DIT_PATH = WEIGHTS_DIR / "hunyuanvideo1.5_720p_t2v_fp16.safetensors"
    VAE_PATH = WEIGHTS_DIR / "hunyuanvideo15_vae_fp16.safetensors"
    TEXT_ENCODER1_PATH = WEIGHTS_DIR / "text_encoder" / "qwen_2.5_vl_7b.safetensors"
    TEXT_ENCODER2_PATH = WEIGHTS_DIR / "text_encoder_2" / "byt5_small_glyphxl_fp16.safetensors"
    MUSUBI_DIR = Path("/src/musubi-tuner")
    
    # Setup directories
    data_dir = Path("/src/training_data")
    output_dir = Path("/src/output")
    cache_dir = data_dir / "cache"
    
    for d in [data_dir, output_dir, cache_dir]:
        d.mkdir(parents=True, exist_ok=True)
    
    print("=== HunyuanVideo 1.5 LoRA Training ===")
    print(f"Resolution: {resolution_width}x{resolution_height}")
    print(f"Steps: {train_steps}, LR: {learning_rate}, Rank: {lora_rank}")
    
    # List musubi-tuner scripts to understand available options
    print("\n[0/4] Checking musubi-tuner scripts...")
    subprocess.run(["ls", "-la", str(MUSUBI_DIR)], check=False)
    subprocess.run(["ls", "-la", str(MUSUBI_DIR / "src" / "musubi_tuner")], check=False)
    
    # Find scripts - try HunyuanVideo 1.5 specific scripts first, fall back to 1.0 scripts
    cache_latent_script = find_script(MUSUBI_DIR, [
        "hv15_cache_latents.py",
        "hunyuan15_cache_latents.py", 
        "hv_1_5_cache_latents.py",
        "cache_latents.py"  # fallback to standard
    ])
    
    cache_te_script = find_script(MUSUBI_DIR, [
        "hv15_cache_text_encoder_outputs.py",
        "hunyuan15_cache_text_encoder_outputs.py",
        "hv_1_5_cache_text_encoder_outputs.py", 
        "cache_text_encoder_outputs.py"  # fallback to standard
    ])
    
    train_script = find_script(MUSUBI_DIR, [
        "hv15_train_network.py",
        "hunyuan15_train_network.py",
        "hv_1_5_train_network.py",
        "hv_train_network.py"  # fallback to standard
    ])
    
    print(f"Cache latent script: {cache_latent_script}")
    print(f"Cache TE script: {cache_te_script}")
    print(f"Train script: {train_script}")
    
    if not all([cache_latent_script, cache_te_script, train_script]):
        raise RuntimeError(f"Could not find required musubi-tuner scripts!")
    
    # Extract and caption
    print("\n[1/4] Extracting and captioning data...")
    data_dir, media_files = extract_and_caption(str(input_videos), data_dir, trigger_word)
    print(f"Found {len(media_files)} media files")
    
    # Create dataset config
    dataset_config = data_dir / "dataset.toml"
    create_dataset_config(str(data_dir), str(dataset_config), resolution_width, resolution_height)
    
    # Cache latents
    print("\n[2/4] Caching latents (HunyuanVideo 1.5)...")
    cache_latent_cmd = [
        "python", str(cache_latent_script),
        "--dataset_config", str(dataset_config),
        "--vae", str(VAE_PATH),
        "--vae_chunk_size", "32",
        "--vae_tiling",
    ]
    subprocess.run(cache_latent_cmd, check=True)
    
    # Cache text encoder outputs
    print("\n[3/4] Caching text encoder outputs (Qwen2.5-VL + ByT5)...")
    cache_te_cmd = [
        "python", str(cache_te_script),
        "--dataset_config", str(dataset_config),
        "--text_encoder1", str(TEXT_ENCODER1_PATH),
        "--text_encoder2", str(TEXT_ENCODER2_PATH),
        "--batch_size", "4",
    ]
    # Add fp8_llm flag if the text encoder supports it
    if "hv15" in str(cache_te_script) or "hunyuan15" in str(cache_te_script):
        cache_te_cmd.append("--fp8_llm")
    subprocess.run(cache_te_cmd, check=True)
    
    # Train
    print("\n[4/4] Training HunyuanVideo 1.5 LoRA...")
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
        "--learning_rate", str(learning_rate),
        "--gradient_checkpointing",
        "--max_data_loader_n_workers", "2",
        "--persistent_data_loader_workers",
        "--network_module", "networks.lora",
        "--network_dim", str(lora_rank),
        "--timestep_sampling", "shift",
        "--discrete_flow_shift", "7.0",
        "--max_train_steps", str(train_steps),
        "--save_every_n_steps", str(max(train_steps // 4, 100)),
        "--seed", "42",
        "--output_dir", str(output_dir),
        "--output_name", "hunyuan15_lora",
        "--blocks_to_swap", str(blocks_to_swap),
    ]
    subprocess.run(train_cmd, check=True)
    
    # Find output
    lora_files = list(output_dir.glob("*.safetensors"))
    if not lora_files:
        raise RuntimeError("No LoRA file generated!")
    
    final_lora = sorted(lora_files, key=lambda x: x.stat().st_mtime)[-1]
    print(f"\n=== Training Complete ===")
    print(f"Output: {final_lora}")
    
    return TrainingOutput(weights=CogPath(final_lora))
