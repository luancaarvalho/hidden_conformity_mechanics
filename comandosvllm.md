Gemma27B 
--- 

CUDA_VISIBLE_DEVICES=0,2 \
vllm serve google/gemma-3-27b-it \
  --served-model-name gemma-3-27b-it \
  --host 0.0.0.0 \
  --port 8000 \
  --tensor-parallel-size 2 \
  --dtype bfloat16 \
  --gpu-memory-utilization 0.98 \
  --max-model-len 8192 \
  --max-num-seqs 64 \
  --max-num-batched-tokens 32768 \
  --enable-chunked-prefill \
  --enable-prefix-caching \
  --swap-space 8 \
  --generation-config vllm

---