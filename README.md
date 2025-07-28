# bert-base-inf-2 — offline batching / on‑demand

**Activate venv**

```bash
source /opt/aws_neuronx_venv_pytorch_2_7_transformers/bin/activate
```

initial:

| Filename                  | Batch | Batches | Inferences | Threads | Models | Duration (s) | Throughput (inf/s) | P50 (s) | P95 (s) | P99 (s) |
| ------------------------- | ----: | ------: | ---------: | ------: | -----: | -----------: | -----------------: | ------: | ------: | ------: |
| model\_batch\_size\_1.pt  |     1 |   12000 |      12000 |      12 |     12 |        0.949 |          12649.572 |   0.939 |   0.974 |   0.990 |
| model\_batch\_size\_2.pt  |     2 |   12000 |      24000 |      12 |     12 |        1.559 |          15394.022 |   1.522 |   1.594 |   1.631 |
| model\_batch\_size\_3.pt  |     3 |   12000 |      36000 |      12 |     12 |        2.134 |          16869.460 |   2.115 |   2.192 |   2.220 |
| model\_batch\_size\_4.pt  |     4 |   12000 |      48000 |      12 |     12 |        2.730 |      **17584.681** |   2.723 |   2.794 |   2.820 |
| model\_batch\_size\_5.pt  |     5 |   12000 |      60000 |      12 |     12 |        3.561 |          16851.301 |   3.512 |   3.659 |   3.694 |
| model\_batch\_size\_6.pt  |     6 |   12000 |      72000 |      12 |     12 |        4.331 |          16624.728 |   4.302 |   4.486 |   4.569 |
| model\_batch\_size\_7.pt  |     7 |   12000 |      84000 |      12 |     12 |        5.018 |          16738.430 |   4.940 |   5.292 |   5.331 |
| model\_batch\_size\_8.pt  |     8 |   12000 |      96000 |      12 |     12 |        5.827 |          16475.001 |   5.760 |   6.305 |   6.426 |
| model\_batch\_size\_9.pt  |     9 |   12000 |     108000 |      12 |     12 |        7.002 |          15423.520 |   6.826 |   7.422 |   7.536 |
| model\_batch\_size\_10.pt |    10 |   12000 |     120000 |      12 |     12 |        7.752 |          15480.798 |   7.577 |   8.229 |   8.504 |

Note: Peak throughput observed at batch size 4 (≈17,585 inf/s); beyond that, throughput flattens or declines while latencies increase.
