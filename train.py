"""
HunyuanVideo 1.5 LoRA Trainer for Replicate
Uses kohya-ss/musubi-tuner with Florence-2 auto-captioning
"""

import os
import sys
import subprocess
import zipfile
import shutil
from pathlib import Path
from cog import BaseModel, Input, Path as CogPath

# Add musubi-tuner to path
sys.path.insert(0, "/src/musubi-tuner")

class TrainingOutput(BaseModel):
    weights: CogPath

def download_weights():
    """Download HunyuanVideo 1.5 weights"""
    import subprocess
    
    weights_dir = Path("/src/weights")
    weights_dir.mkdir(parents=True, exist_ok=True)
    
    # DiT model
    dit_path = weights_dir / "hunyuan-video-1.5-dit.pt"
    if not dit_path.exists():
        print("Downloading HunyuanVideo 1.5 DiT...")
        subprocess.run([
            "pget", "-x",
            "https://huggingface.co/tencent/HunyuanVideo-1.5/resolve/main/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states.pt",
            str(dit_path)
        ], check=True)
    
    # VAE
    vae_path = weights_dir / "hunyuan-video-vae.pt"
    if not vae_path.exists():
        print("Downloading VAE...")
        subprocess.run([
            "pget", "-x",
            "https://huggingface.co/tencent/HunyuanVideo/resolve/main/hunyuan-video-t2v-720p/vae/pytorch_model.pt",
            str(vae_path)
        ], check=True)
    
    return weights_dir

def setup_florence2():
    """Setup Florence-2 for auto-captioning"""
    from transformers import AutoProcessor, AutoModelForCausalLM
    import torch
    
    model_id = "microsoft/Florence-2-large"
    
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        trust_remote_code=True
    ).cuda()
    
    return processor, model

def caption_image(image_path, processor, model, trigger_word):
    """Generate caption for an image using Florence-2"""
    from PIL import Image
    import torch
    
    image = Image.open(image_path).convert("RGB")
    
    prompt = "&lt;MORE_DETAILED_CAPTION&gt;"
    inputs = processor(text=prompt, images=image, return_tensors="pt").to("cuda", torch.float16)
    
    generated_ids = model.generate(
        input_ids=inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        max_new_tokens=256,
        num_beams=3
    )
    
    caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    caption = caption.replace("&lt;MORE_DETAILED_CAPTION&gt;", "").strip()
    
    # Add trigger word
    return f"{trigger_word}, {caption}"

def extract_and_caption(zip_path, output_dir, trigger_word):
    """Extract ZIP and auto-caption files without captions"""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract ZIP
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(output_dir)
    
    # Find media files
    image_exts = {'.jpg', '.jpeg', '.png', '.webp'}
    video_exts = {'.mp4', '.mov', '.webm'}
    
    media_files = []
    for f in output_dir.rglob("*"):
        if f.suffix.lower() in image_exts | video_exts:
            media_files.append(f)
    
    # Check which files need captions
    needs_caption = []
    for media in media_files:
        caption_file = media.with_suffix('.txt')
        if not caption_file.exists():
            needs_caption.append(media)
    
    if needs_caption:
        print(f"Auto-captioning {len(needs_caption)} files with Florence-2...")
        processor, model = setup_florence2()
        
        for media in needs_caption:
            if media.suffix.lower() in image_exts:
                caption = caption_image(media, processor, model, trigger_word)
            else:
                # For videos, caption first frame
                import cv2
                cap = cv2.VideoCapture(str(media))
                ret, frame = cap.read()
                cap.release()
                
                if ret:
                    temp_img = output_dir / "temp_frame.jpg"
                    cv2.imwrite(str(temp_img), frame)
                    caption = caption_image(temp_img, processor, model, trigger_word)
                    temp_img.unlink()
                else:
                    caption = f"{trigger_word}, a video"
            
            caption_file = media.with_suffix('.txt')
            caption_file.write_text(caption)
            print(f"  {media.name}: {caption[:80]}...")
        
        # Clean up Florence-2
        del processor, model
        import torch
        torch.cuda.empty_cache()
    
    return output_dir, media_files

def create_dataset_config(data_dir, output_path, resolution_width, resolution_height):
    """Create TOML dataset config for musubi-tuner"""
    
    config = f'''[general]
shuffle_caption = true
caption_extension = ".txt"
keep_tokens = 1

[[datasets]]
resolution = [{resolution_width}, {resolution_height}]
batch_size = 1

[[datasets.subsets]]
image_dir = "{data_dir}"
num_repeats = 1
'''
    
    Path(output_path).write_text(config)
    return output_path

def train(
    input_data: CogPath = Input(description="ZIP file with images/videos (and optional .txt captions)"),
    trigger_word: str = Input(description="Trigger word for your LoRA", default="TOK"),
    epochs: int = Input(description="Number of training epochs", default=10, ge=1, le=100),
    lora_rank: int = Input(description="LoRA rank (dimension)", default=32, ge=4, le=128),
    learning_rate: float = Input(description="Learning rate", default=2e-4, ge=1e-6, le=1e-2),
    resolution_width: int = Input(description="Training resolution width", default=544, ge=256, le=1280),
    resolution_height: int = Input(description="Training resolution height", default=960, ge=256, le=1280),
    use_fp8: bool = Input(description="Use FP8 quantization (saves VRAM)", default=True),
    blocks_to_swap: int = Input(description="Blocks to swap to CPU (0-36, higher = less VRAM)", default=32, ge=0, le=36),
    gradient_checkpointing: bool = Input(description="Enable gradient checkpointing", default=True),
) -> TrainingOutput:
    """Train a HunyuanVideo 1.5 LoRA"""
    
    print("=" * 60)
    print("HunyuanVideo 1.5 LoRA Trainer")
    print("=" * 60)
    
    # Setup directories
    work_dir = Path("/src/training")
    work_dir.mkdir(parents=True, exist_ok=True)
    
    data_dir = work_dir / "data"
    output_dir = work_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Download model weights
    print("\n[1/5] Downloading model weights...")
    weights_dir = download_weights()
    
    # Extract and caption data
    print("\n[2/5] Preparing training data...")
    data_dir, media_files = extract_and_caption(input_data, data_dir, trigger_word)
    print(f"Found {len(media_files)} media files")
    
    # Create dataset config
    print("\n[3/5] Creating dataset configuration...")
    dataset_config = work_dir / "dataset.toml"
    create_dataset_config(str(data_dir), str(dataset_config), resolution_width, resolution_height)
    
    # Cache latents
    print("\n[4/5] Caching latents...")
    cache_cmd = [
        "python", "/src/musubi-tuner/cache_latents.py",
        "--dataset_config", str(dataset_config),
        "--vae", str(weights_dir / "hunyuan-video-vae.pt"),
        "--vae_chunk_size", "32",
        "--vae_tiling",
    ]
    subprocess.run(cache_cmd, check=True)
    
    # Cache text encoder outputs
    print("Caching text encoder outputs...")
    cache_te_cmd = [
        "python", "/src/musubi-tuner/cache_text_encoder_outputs.py",
        "--dataset_config", str(dataset_config),
        "--text_encoder1", "Qwen/Qwen2.5-VL-7B-Instruct",
        "--text_encoder2", "google/byt5-small",
        "--batch_size", "1",
    ]
    subprocess.run(cache_te_cmd, check=True)
    
    # Train
    print("\n[5/5] Training LoRA...")
    train_cmd = [
        "python", "/src/musubi-tuner/hv_train_network.py",
        "--dataset_config", str(dataset_config),
        "--dit", str(weights_dir / "hunyuan-video-1.5-dit.pt"),
        "--network_module", "networks.lora",
        "--network_dim", str(lora_rank),
        "--network_alpha", str(lora_rank // 2),
        "--optimizer_type", "adamw8bit",
        "--learning_rate", str(learning_rate),
        "--max_train_epochs", str(epochs),
        "--save_every_n_epochs", str(max(1, epochs // 3)),
        "--output_dir", str(output_dir),
        "--output_name", f"hunyuan15_lora_{trigger_word}",
        "--timestep_sampling", "sigmoid",
        "--discrete_flow_shift", "1.0",
        "--max_data_loader_n_workers", "2",
        "--persistent_data_loader_workers",
        "--mixed_precision", "bf16",
        "--seed", "42",
    ]
    
    if use_fp8:
        train_cmd.extend(["--fp8_base"])
    
    if blocks_to_swap > 0:
        train_cmd.extend(["--blocks_to_swap", str(blocks_to_swap)])
    
    if gradient_checkpointing:
        train_cmd.extend(["--gradient_checkpointing"])
    
    subprocess.run(train_cmd, check=True)
    
    # Find output weights
    output_files = list(output_dir.glob("*.safetensors"))
    if not output_files:
        raise RuntimeError("Training completed but no output weights found")
    
    # Get the final weights (highest epoch or last saved)
    final_weights = sorted(output_files)[-1]
    
    print(f"\n{'=' * 60}")
    print(f"Training complete! Output: {final_weights.name}")
    print(f"{'=' * 60}")
    
    return TrainingOutput(weights=CogPath(final_weights))
