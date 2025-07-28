# bert-inf-2 - offline batching/ on demand

activate venv- source /opt/aws_neuronx_venv_pytorch_2_7_transformers//bin/activate

initial:

------------------------------------------------------------
Filename:    model_batch_size_1.pt
Batch Size:  1
Batches:     12000
Inferences:  12000
Threads:     12
Models:      12
Duration:    0.949
Throughput:  12649.572
Latency P50: 0.939
Latency P95: 0.974
Latency P99: 0.990

------------------------------------------------------------
Filename:    model_batch_size_2.pt
Batch Size:  2
Batches:     12000
Inferences:  24000
Threads:     12
Models:      12
Duration:    1.559
Throughput:  15394.022
Latency P50: 1.522
Latency P95: 1.594
Latency P99: 1.631

------------------------------------------------------------
Filename:    model_batch_size_3.pt
Batch Size:  3
Batches:     12000
Inferences:  36000
Threads:     12
Models:      12
Duration:    2.134
Throughput:  16869.460
Latency P50: 2.115
Latency P95: 2.192
Latency P99: 2.220

------------------------------------------------------------
Filename:    model_batch_size_4.pt
Batch Size:  4
Batches:     12000
Inferences:  48000
Threads:     12
Models:      12
Duration:    2.730
Throughput:  17584.681
Latency P50: 2.723
Latency P95: 2.794
Latency P99: 2.820

------------------------------------------------------------
Filename:    model_batch_size_5.pt
Batch Size:  5
Batches:     12000
Inferences:  60000
Threads:     12
Models:      12
Duration:    3.561
Throughput:  16851.301
Latency P50: 3.512
Latency P95: 3.659
Latency P99: 3.694

------------------------------------------------------------
Filename:    model_batch_size_6.pt
Batch Size:  6
Batches:     12000
Inferences:  72000
Threads:     12
Models:      12
Duration:    4.331
Throughput:  16624.728
Latency P50: 4.302
Latency P95: 4.486
Latency P99: 4.569

------------------------------------------------------------
Filename:    model_batch_size_7.pt
Batch Size:  7
Batches:     12000
Inferences:  84000
Threads:     12
Models:      12
Duration:    5.018
Throughput:  16738.430
Latency P50: 4.940
Latency P95: 5.292
Latency P99: 5.331

------------------------------------------------------------
Filename:    model_batch_size_8.pt
Batch Size:  8
Batches:     12000
Inferences:  96000
Threads:     12
Models:      12
Duration:    5.827
Throughput:  16475.001
Latency P50: 5.760
Latency P95: 6.305
Latency P99: 6.426

------------------------------------------------------------
Filename:    model_batch_size_9.pt
Batch Size:  9
Batches:     12000
Inferences:  108000
Threads:     12
Models:      12
Duration:    7.002
Throughput:  15423.520
Latency P50: 6.826
Latency P95: 7.422
Latency P99: 7.536

------------------------------------------------------------
Filename:    model_batch_size_10.pt
Batch Size:  10
Batches:     12000
Inferences:  120000
Threads:     12
Models:      12
Duration:    7.752
Throughput:  15480.798
Latency P50: 7.577
Latency P95: 8.229
Latency P99: 8.504