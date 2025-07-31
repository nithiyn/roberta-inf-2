# bert-base-inf-2 — offline batching / on‑demand

**Activate venv**

```bash
source /opt/aws_neuronx_venv_pytorch_2_7_transformers/bin/activate
```

initial:


| Filename | Batch Size | Batches | Inferences | Threads | Models | Duration | Throughput | Latency P50 | Latency P95 | Latency P99 |
|----------|------------|---------|------------|---------|--------|----------|------------|-------------|-------------|-------------|
| model_256_batch.pt | 256 | 1 | 256 | 1 | 1 | 0.708 | 361.378 | 707.116 | 707.116 | 707.116 |
| model_128_batch.pt | 128 | 1 | 128 | 1 | 1 | 0.355 | 360.091 | 354.756 | 354.756 | 354.756 |
| model_64_batch.pt | 64 | 1 | 64 | 1 | 1 | 0.179 | 357.496 | 178.184 | 178.184 | 178.184 |
| model_32_batch.pt | 32 | 1 | 32 | 1 | 1 | 0.091 | 352.749 | 89.927 | 89.927 | 89.927 |
| model_16_batch.pt | 16 | 1 | 16 | 1 | 1 | 0.046 | 347.588 | 45.187 | 45.187 | 45.187 |
| model_8_batch.pt | 8 | 1 | 8 | 1 | 1 | 0.023 | 342.480 | 22.590 | 22.590 | 22.590 |
| model_4_batch.pt | 4 | 1 | 4 | 1 | 1 | 0.013 | 308.274 | 12.399 | 12.399 | 12.399 |
