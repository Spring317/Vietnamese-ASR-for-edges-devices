# Vietnamese ASR for Edge Devices

A lightweight Automatic Speech Recognition model for Vietnamese language that can be deployed on resource-constrained edge devices.

## Introduction

This project aims to create an efficient ASR solution for Vietnamese that works well on edge devices with limited computational resources. The model is optimized through quantization techniques to reduce size and improve inference speed while maintaining acceptable accuracy.

## Installation

### Prerequisites
- Python 3.8 or higher
- Transformers
- Vitis-AI v3.0 
- TVM v0.19.0
  
# Clone the repository
git clone https://github.com/yourusername/Vietnamese-ASR-for-edges-devices.git
cd Vietnamese-ASR-for-edges-devices

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Running the Model
```bash
# Example command to run inference
python run_inference.py --audio_file path/to/audio.wav --model_type dynamic
```

### Compiling for Edge Devices
```bash
# Example command for quantization
python quantize_model.py --config avx2 --type dynamic
```

## Results

Performance metrics for different quantization approaches:

| Quantization Method | Word Error Rate (WER) | Model Size Reduction |
|---------------------|:---------------------:|:--------------------:|
| Dynamic quantize    | 11%                   | TBD                  |
| Static quantize     | 30%                   | TBD                  |
| Original model      | TBD                   | N/A                  |

## Roadmap

- [x] Quantize model with Huggingface.Optimum framework using AVX2 quantize config
  - [x] Dynamic quantization (WER = 11%)
  - [x] Static quantization (WER = 30%)
- [ ] Deploy to AMD KV260 kit (ONNX Runtime)
- [ ] Explore using Vitis AI for further optimization