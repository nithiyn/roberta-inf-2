# MODIFIED VERSION - batch_plateau_test.py

import logging
import torch
import torch_neuronx
from benchmark import simple_benchmark
from bert_encode import BERTEncoder

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    datefmt='%H:%M:%S'
)

def main():
    logging.info("🚀 Starting BERT batch plateau testing...")
    
    # Initialize encoder ONCE (more efficient)
    logging.info("📥 Initializing BERTEncoder...")
    encoder = BERTEncoder(model_name="bert-base-cased-finetuned-mrpc")
    logging.info("✅ Encoder initialized")
    
    # Test batch sizes to find plateau
    batch_sizes_to_test = [4, 8, 16, 32, 64, 128, 256]
    compiled_models = []
    
    # Compile BERT for different batch sizes
    logging.info("🔧 Starting model compilation for batch plateau testing...")
    
    for batch_size in batch_sizes_to_test:
        try:
            logging.info(f"⚡ Compiling model for batch size {batch_size}...")
            
            # Use smaller sequence length for larger batches to avoid compilation issues
            max_length = 128 if batch_size <= 32 else 64 if batch_size <= 128 else 32
            logging.info(f"   Using max_length={max_length} for batch_size={batch_size}")
            
            # Create example with appropriate sequence length
            example = encoder.encode(encoder.sequence_0, encoder.sequence_2, 
                                   max_length=max_length, batch_size=batch_size)
            
            # Trace using the encoder's model
            model_neuron = torch_neuronx.trace(encoder.model, example)
            
            # Save the traced model
            filename = f'model_batch_size_{batch_size}.pt'
            torch.jit.save(model_neuron, filename)
            logging.info(f"✅ Saved {filename}")
            compiled_models.append((batch_size, filename, max_length))
            
        except Exception as e:
            logging.error(f"❌ Compilation failed for batch size {batch_size}: {e}")
            logging.info(f"   Stopping compilation at batch size {batch_size}")
            break
    
    if not compiled_models:
        logging.error("❌ No models compiled successfully!")
        return
    
    logging.info(f"🎯 Successfully compiled {len(compiled_models)} models")
    
    # Benchmark BERT for compiled batch sizes (single batch inference only)
    logging.info("📊 Starting single-batch benchmarking...")
    
    results = []
    
    for batch_size, filename, max_length in compiled_models:
        try:
            print('='*70)
            logging.info(f"🏃 Benchmarking batch size {batch_size} (max_length={max_length})...")
            
            # Create example for this batch size with same max_length used in compilation
            example = encoder.encode(encoder.sequence_0, encoder.sequence_2, 
                                   max_length=max_length, batch_size=batch_size)
            
            # Run single-batch benchmark (modified parameters)
            metrics = simple_benchmark(
                filename=filename, 
                example=example,     # Single model
                #n_threads=1,       # Single thread  
                #batches_per_thread=1  # Single batch inference
            )
            
            # Store results for plateau analysis
            results.append({
                'batch_size': batch_size,
                'max_length': max_length,
                'throughput': metrics['throughput'],
                'latency_ms': metrics['latency_p50'],
                'inferences_per_batch': batch_size
            })
            
            # Log key metrics
            logging.info(f"✅ Batch {batch_size}: {metrics['throughput']:.2f} inferences/sec, "
                        f"Latency: {metrics['latency_p50']:.2f}ms")
            print()
            
        except Exception as e:
            logging.error(f"❌ Benchmark failed for batch size {batch_size}: {e}")
            continue
    
    # Analyze plateau
    logging.info("📈 PLATEAU ANALYSIS:")
    print("\n" + "="*80)
    print("BATCH SIZE PLATEAU ANALYSIS")
    print("="*80)
    print(f"{'Batch Size':<12} {'Max Length':<12} {'Throughput':<15} {'Latency (ms)':<15} {'Efficiency':<12}")
    print("-"*80)
    
    max_throughput = max(r['throughput'] for r in results) if results else 0
    
    for result in results:
        efficiency = (result['throughput'] / max_throughput) * 100 if max_throughput > 0 else 0
        print(f"{result['batch_size']:<12} {result['max_length']:<12} "
              f"{result['throughput']:<15.2f} {result['latency_ms']:<15.2f} {efficiency:<12.1f}%")
    
    # Find optimal batch size
    if results:
        optimal = max(results, key=lambda x: x['throughput'])
        logging.info(f"🎯 OPTIMAL BATCH SIZE: {optimal['batch_size']} "
                    f"({optimal['throughput']:.2f} inferences/sec)")
        
        # Check for plateau (when throughput improvement < 5%)
        plateau_threshold = 0.05  # 5% improvement threshold
        plateau_found = False
        
        for i in range(1, len(results)):
            prev_throughput = results[i-1]['throughput']
            curr_throughput = results[i]['throughput']
            improvement = (curr_throughput - prev_throughput) / prev_throughput
            
            if improvement < plateau_threshold:
                logging.info(f"📊 PLATEAU DETECTED: Performance improvement drops below 5% "
                           f"after batch size {results[i-1]['batch_size']}")
                plateau_found = True
                break
        
        if not plateau_found:
            logging.info("📊 No clear plateau detected in tested range. "
                        "Consider testing larger batch sizes.")
    
    logging.info("🎉 Batch plateau testing completed!")

if __name__ == "__main__":
    main()