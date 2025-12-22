# HunyuanVideo 1.5 LoRA Trainer (Local Docker)

Train HunyuanVideo 1.5 LoRAs locally using musubi-tuner in Docker.

Based on official musubi-tuner docs: https://github.com/kohya-ss/musubi-tuner/blob/main/docs/hunyuan_video_1_5.md

## Requirements

- NVIDIA GPU with 24GB+ VRAM (3090, 4090, A100)
- Docker with nvidia-container-toolkit
- ~50GB disk space for model weights (cached after first run)

## Quick Start

```bash
# 1. Put your training data in ./data/
cp /path/to/your-dataset.zip data/

# 2. Build the container
docker-compose build train

# 3. Run training
docker-compose run train

# Or run quick test (1 epoch)
docker-compose run test

# Or debug interactively
docker-compose run shell
```

## Dataset Format

ZIP file with images/videos + matching .txt caption files:

```
mydata.zip
├── image001.png
├── image001.txt    # "a beautiful sunset over mountains"
├── image002.jpg
├── image002.txt    # "close up of a cat sleeping"
```

**Supported media:**
- Images: .png, .jpg, .jpeg, .webp, .bmp
- Videos: .mp4, .mov, .webm, .mkv, .avi

## Training Parameters

Edit `docker-compose.yml` or run directly:

```bash
docker-compose run shell
python train_local.py \
    --input_zip /src/input/zip/mydata.zip \
    --task t2v \
    --trigger_word "mystyle" \
    --epochs 10 \
    --rank 32 \
    --learning_rate 1e-4 \
    --fp8
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--input_zip` | required | ZIP with images/videos + .txt captions |
| `--task` | t2v | `t2v` (text-to-video) or `i2v` (image-to-video) |
| `--trigger_word` | "" | Prepended to all captions |
| `--epochs` | 10 | Training epochs |
| `--max_train_steps` | -1 | Override epochs with step count |
| `--rank` | 32 | LoRA dimension (4-128) |
| `--learning_rate` | 1e-4 | Learning rate |
| `--optimizer` | adamw8bit | adamw8bit, adamw, AdaFactor |
| `--fp8` | false | Use FP8 for DiT (saves VRAM on 24GB cards) |
| `--skip_download` | false | Skip model download (if already cached) |
| `--skip_cache` | false | Skip latent caching (if already cached) |

## Output

After training:
- `output/lora.safetensors` - musubi-tuner format
- `output/lora_comfyui.safetensors` - ComfyUI format

## Memory Usage (3090 24GB)

With `--fp8` flag:
- Model loading: ~12GB
- Training: ~20-22GB peak
- Safe margin for 24GB card

Without `--fp8`:
- May OOM on 24GB cards
- Recommended for 48GB+ cards only

## Troubleshooting

**OOM during training:**
```bash
# Use FP8 mode
python train_local.py ... --fp8

# Or reduce batch size in train.toml (already set to 1)
```

**Download failures:**
Models are large (~25GB total). If download fails:
```bash
# Resume with skip flag
python train_local.py ... --skip_download
```

**Resume after cache:**
If caching completed but training failed:
```bash
python train_local.py ... --skip_download --skip_cache
```

## Model Weights

First run downloads ~25GB of weights to `./models/`:
- DiT (T2V or I2V): ~13GB
- VAE: ~500MB
- Text Encoder (Qwen2.5-VL): ~8GB
- BYT5: ~200MB
- Image Encoder (I2V only): ~400MB

These are cached and reused for subsequent runs.

## Architecture

```
train_local.py
├── download_weights()     # Get models from HuggingFace
├── extract_zip()          # Unpack training data
├── create_dataset_toml()  # Generate musubi-tuner config
├── cache_latents()        # Pre-compute VAE latents
├── cache_text_encoder()   # Pre-compute text embeddings
├── run_training()         # Actual LoRA training
└── convert_to_comfyui()   # Convert output format
```

Uses official musubi-tuner scripts:
- `hv_1_5_cache_latents.py`
- `hv_1_5_cache_text_encoder_outputs.py`
- `hv_1_5_train_network.py`
