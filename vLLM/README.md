Конфигурация для карты NVIDIA RTX PRO 4500 32Gb

```bash exec="1" source="block"
docker run --rm --gpus all -p 8000:8000 --name vLLM --shm-size 16g \
  -v /var/models:/models:Z \
  -e HF_HUB_OFFLINE=1 \
  vllm/vllm-openai:latest \
  /models/Qwen3.5-35B-A3B-AWQ-4bit \
  --kv-cache-dtype fp8_e4m3 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.9859  \
  --limit-mm-per-prompt '{"image": 2}' \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder \
  --served-model-name Qwen3.5-35B-A3B-AWQ \
  --default-chat-template-kwargs '{"enable_thinking": false}' \
  --reasoning-parser qwen3
```
