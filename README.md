# RoBERTa NER Inference on AWS Neuron — Batch Optimization & Performance investigation

This project benchmarks **Named Entity Recognition (NER)** using RoBERTa models on inf2, focusing on batch size optimization and sequence length analysis for maximum throughput and latency efficiency.

## Quick Start

**Activate AWS Neuron Environment**
```bash
source /opt/aws_neuronx_venv_pytorch_2_7_nxd_inference/bin/activate
```

**Install Dependencies**
```bash
pip install -r requirements.txt
```

## Project Structure

```
src/
├── bert_encode.py           # RoBERTa NER encoder class
├── one_core_test.py         # Single core NER tes- logits and output
├── multi_batch_test.py      # Multi-batch size benchmarking (128 batches each)
            
```

## Available Test Scripts

### 1. Single Core Test
```bash
python src/one_core_test.py
```
Basic NER functionality test with model compilation and inference validation.

### 2. Multi-Batch Size Benchmark
```bash
python src/multi_batch_test.py
```
compile+  benchmarking across batch sizes [4, 8, 16, 32] with 128 batches per size.


### Batch Size Performance Analysis -roberta-base
| Batch Size | Seq Len | Filename                           | Batches | Inferences | Threads | Models | Duration (s) | Throughput (inf/s) | P50 (ms) | P95 (ms) | P99 (ms) |
| ---------: | ------: | ---------------------------------- | ------: | ---------: | ------: | -----: | -----------: | -----------------: | -------: | -------: | -------: |
|          4 |     128 | ner\_model\_batch\_4\_seq\_128.pt  |     128 |        512 |       1 |      1 |        0.352 |            1453.65 |    2.753 |    2.763 |    2.783 |
|            |     512 | ner\_model\_batch\_4\_seq\_512.pt  |     128 |        512 |       1 |      1 |        1.889 |             271.04 |   14.755 |   14.796 |   14.808 |
|          8 |     128 | ner\_model\_batch\_8\_seq\_128.pt  |     128 |       1024 |       1 |      1 |        0.687 |            1490.18 |    5.365 |    5.394 |    5.405 |
|            |     512 | ner\_model\_batch\_8\_seq\_512.pt  |     128 |       1024 |       1 |      1 |        3.464 |             295.63 |   27.060 |   27.073 |   27.077 |
|         16 |     128 | ner\_model\_batch\_16\_seq\_128.pt |     128 |       2048 |       1 |      1 |        1.384 |            1479.79 |   10.814 |   10.821 |   10.836 |
|            |     512 | ner\_model\_batch\_16\_seq\_512.pt |     128 |       2048 |       1 |      1 |        6.932 |             295.42 |   54.131 |   54.194 |   54.303 |
|         32 |     128 | ner\_model\_batch\_32\_seq\_128.pt |     128 |       4096 |       1 |      1 |        2.777 |            1474.86 |   21.690 |   21.749 |   21.809 |
|            |     512 | ner\_model\_batch\_32\_seq\_512.pt |     128 |       4096 |       1 |      1 |       13.817 |             296.44 |  107.938 |  107.999 |  108.041 |


### Batch Size prrf- roberta-large
| Batch Size | Seq Len | Filename                                  | Batches | Inferences | Threads | Models | Duration (s) | Throughput (inf/s) | P50 (ms) | P95 (ms) | P99 (ms) |
| ---------: | ------: | ----------------------------------------- | ------: | ---------: | ------: | -----: | -----------: | -----------------: | -------: | -------: | -------: |
|          4 |     128 | ner\_large\_model\_batch\_4\_seq\_128.pt  |     128 |        512 |       1 |      1 |        1.088 |             470.76 |    8.497 |    8.516 |    8.540 |
|            |     512 | ner\_large\_model\_batch\_4\_seq\_512.pt  |     128 |        512 |       1 |      1 |        5.515 |              92.84 |   43.067 |   43.145 |   43.194 |
|          8 |     128 | ner\_large\_model\_batch\_8\_seq\_128.pt  |     128 |       1024 |       1 |      1 |        2.209 |             463.53 |   17.247 |   17.325 |   17.349 |
|            |     512 | ner\_large\_model\_batch\_8\_seq\_512.pt  |     128 |       1024 |       1 |      1 |       10.635 |              96.29 |   83.063 |   83.125 |   83.147 |
|         16 |     128 | ner\_large\_model\_batch\_16\_seq\_128.pt |     128 |       2048 |       1 |      1 |        4.355 |             470.29 |   34.002 |   34.061 |   34.071 |
|            |     512 | ner\_large\_model\_batch\_16\_seq\_512.pt |     128 |       2048 |       1 |      1 |       21.961 |              93.26 |  171.460 |  171.530 |  175.735 |
|         32 |     128 | ner\_large\_model\_batch\_32\_seq\_128.pt |     128 |       4096 |       1 |      1 |        8.888 |             460.84 |   69.426 |   69.462 |   69.489 |
|            |     512 | ner\_large\_model\_batch\_32\_seq\_512.pt |     128 |       4096 |       1 |      1 |       43.095 |              95.05 |  336.673 |  336.781 |  336.870 |




##  Configuration Options

### Environment Variables
```bash
export NEURON_CC_FLAGS="--model-type=transformer --dump=../ncc_dump --cache_dir=./neuron_cache"
export NEURON_FRAMEWORK_DEBUG="1"
```

### NER Entity Detection Example
```
--- Testing text 1: The company HuggingFace is based in New York City ---
Found entities:
  HuggingFace -> B-ORG (confidence: 0.998)
  New -> B-LOC (confidence: 0.995)
  York -> I-LOC (confidence: 0.994)
  City -> I-LOC (confidence: 0.992)
```

## Development & Debugging

### Compilation Artifacts
- **Neuron Cache**: `./neuron_cache/` - Compiled model cache
- **Compiler Workdir**: `./compiler_workdir/` - Intermediate compilation files
- **NCC Dump**: `../ncc_dump/` - Neuron compiler debug output

### Monitoring Commands
```bash
# Monitor Neuron core utilization
neuron-monitor

# Check Neuron runtime status  
neuron-ls

# View compilation logs
cat src/log-neuron-cc.txt
```

---

*Last updated: August 2025 | AWS Neuron SDK 2.7 | PyTorch 2.7*
