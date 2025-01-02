# Vietnamese-ASR-for-edges-devices
A lightweight model which could be deploy on edges devices
# Done: 
## Quantized model with Huggingface.Optimum framework using avx2 quantize config:
### Dynamic quantize: WER = 11%
### Static quantize: WER = 30%

# TODO
## Optimize with TVM accelerator (IDK I really dont want to re-implement the whole wav2vec2 model)
## Make it real-time (More suffering tbh)
