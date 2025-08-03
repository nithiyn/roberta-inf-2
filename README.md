## bert-base-inf-2 — offline batching / on‑demand

**Activate venv**

```bash
source /opt/aws_neuronx_venv_pytorch_2_7_transformers/bin/activate
```
1 batch per thread

| Filename | Batch Size | Batches | Inferences | Threads | Models | Duration | Throughput | Latency P50 | Latency P95 | Latency P99 |
|----------|------------|---------|------------|---------|--------|----------|------------|-------------|-------------|-------------|
| model_256_batch.pt | 256 | 1 | 256 | 1 | 1 | 0.708 | 361.378 | 707.116 | 707.116 | 707.116 |
| model_128_batch.pt | 128 | 1 | 128 | 1 | 1 | 0.355 | 360.091 | 354.756 | 354.756 | 354.756 |
| model_64_batch.pt | 64 | 1 | 64 | 1 | 1 | 0.179 | 357.496 | 178.184 | 178.184 | 178.184 |
| model_32_batch.pt | 32 | 1 | 32 | 1 | 1 | 0.091 | 352.749 | 89.927 | 89.927 | 89.927 |
| model_16_batch.pt | 16 | 1 | 16 | 1 | 1 | 0.046 | 347.588 | 45.187 | 45.187 | 45.187 |
| model_8_batch.pt | 8 | 1 | 8 | 1 | 1 | 0.023 | 342.480 | 22.590 | 22.590 | 22.590 |
| model_4_batch.pt | 4 | 1 | 4 | 1 | 1 | 0.013 | 308.274 | 12.399 | 12.399 | 12.399 |

### per core sweet spot for model

used neuron-monitor to capture, its counter is per capture window,

batch size 8
```json
"neuroncore_counters":{"period":1.000385359,"neuroncores_in_use":{"0":{"neuroncore_utilization":99.38037580375642,"effective_flops":34776185139834}
```

batch size 16 
```json
"neuroncore_counters":{"period":1.000075097,"neuroncores_in_use":{"0":{"neuroncore_utilization":100,"effective_flops":34553733110090}
```

batch size 32 , seq len 512
 looks like the saturation point, core util is 100%.
 sweet spot seems to be between 8-16 per core batch size

```json
{"neuroncore_utilization":100,"effective_flops":34576410132558},"1",
{"period":1.000580757,"neuroncores_in_use":{"0":{"neuroncore_utilization":100,"effective_flops":34576410132558},"1" 
```
sweet spot for throughput/latency, hardware util is with per core batch = 8–16 for seq len 512. less than or equal to 95% of peak throughput with much lower latency than batch=32 of 90ms, which is where we touch the compute ceiling. for scaling QPS we can do DP with continous batching and have max batch delay at 5-15% of the batch latency we pick between 8-16 for the batch size and look at how we can keep p50/p95 reasonable.
