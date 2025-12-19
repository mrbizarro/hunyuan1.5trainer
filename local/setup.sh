#!/bin/bash
# Local HunyuanVideo 1.5 Training Setup

set -e

WEIGHTS_DIR="./weights"
mkdir -p "$WEIGHTS_DIR/text_encoder" "$WEIGHTS_DIR/text_encoder_2"

echo "=== Downloading HunyuanVideo 1.5 Weights ==="

# DiT model (~25GB)
if [ ! -f "$WEIGHTS_DIR/hunyuan1.5_dit.safetensors" ]; then
    echo "Downloading DiT model..."
    aria2c -x 16 -s 16 -k 1M -o hunyuan1.5_dit.safetensors -d "$WEIGHTS_DIR" \
        "https://huggingface.co/Comfy-Org/HunyuanVideo_1.5_repackaged/resolve/main/split_files/diffusion_models/hunyuanvideo1.5_720p_t2v_fp16.safetensors" \
        || wget -O "$WEIGHTS_DIR/hunyuan1.5_dit.safetensors" \
        "https://huggingface.co/Comfy-Org/HunyuanVideo_1.5_repackaged/resolve/main/split_files/diffusion_models/hunyuanvideo1.5_720p_t2v_fp16.safetensors"
fi

# VAE
if [ ! -f "$WEIGHTS_DIR/hunyuan1.5_vae.safetensors" ]; then
    echo "Downloading VAE..."
    aria2c -x 16 -s 16 -k 1M -o hunyuan1.5_vae.safetensors -d "$WEIGHTS_DIR" \
        "https://huggingface.co/Comfy-Org/HunyuanVideo_1.5_repackaged/resolve/main/split_files/vae/hunyuanvideo15_vae_fp16.safetensors" \
        || wget -O "$WEIGHTS_DIR/hunyuan1.5_vae.safetensors" \
        "https://huggingface.co/Comfy-Org/HunyuanVideo_1.5_repackaged/resolve/main/split_files/vae/hunyuanvideo15_vae_fp16.safetensors"
fi

# Qwen2.5-VL text encoder
if [ ! -f "$WEIGHTS_DIR/text_encoder/qwen_2.5_vl_7b.safetensors" ]; then
    echo "Downloading Qwen2.5-VL text encoder..."
    aria2c -x 16 -s 16 -k 1M -o qwen_2.5_vl_7b.safetensors -d "$WEIGHTS_DIR/text_encoder" \
        "https://huggingface.co/Comfy-Org/HunyuanVideo_1.5_repackaged/resolve/main/split_files/text_encoders/qwen_2.5_vl_7b.safetensors" \
        || wget -O "$WEIGHTS_DIR/text_encoder/qwen_2.5_vl_7b.safetensors" \
        "https://huggingface.co/Comfy-Org/HunyuanVideo_1.5_repackaged/resolve/main/split_files/text_encoders/qwen_2.5_vl_7b.safetensors"
fi

# ByT5 text encoder
if [ ! -f "$WEIGHTS_DIR/text_encoder_2/byt5.safetensors" ]; then
    echo "Downloading ByT5 text encoder..."
    aria2c -x 16 -s 16 -k 1M -o byt5.safetensors -d "$WEIGHTS_DIR/text_encoder_2" \
        "https://huggingface.co/Comfy-Org/HunyuanVideo_1.5_repackaged/resolve/main/split_files/text_encoders/byt5_small_glyphxl_fp16.safetensors" \
        || wget -O "$WEIGHTS_DIR/text_encoder_2/byt5.safetensors" \
        "https://huggingface.co/Comfy-Org/HunyuanVideo_1.5_repackaged/resolve/main/split_files/text_encoders/byt5_small_glyphxl_fp16.safetensors"
fi

# Clone musubi-tuner
if [ ! -d "musubi-tuner" ]; then
    echo "Cloning musubi-tuner..."
    git clone https://github.com/kohya-ss/musubi-tuner.git
fi

echo ""
echo "=== Setup Complete ==="
echo "Weights downloaded to: $WEIGHTS_DIR"
echo "Next: Run train.py with your data"
