import os
import logging
import torch
import torch_neuronx
import time
import json
import concurrent.futures
import threading
from bert_encode import RoBERTaNEREncoder

# Set up environment variables for Neuron compilation
os.environ['NEURON_CC_FLAGS'] = "--model-type=transformer --dump=../ncc_dump --cache_dir=./neuron_cache"
os.environ["NEURON_FRAMEWORK_DEBUG"] = "1"

# Create directories
os.makedirs("../ncc_dump", exist_ok=True)
os.makedirs("./compiler_workdir", exist_ok=True)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    datefmt='%H:%M:%S'
)

def load_or_compile_model_for_batch_size(batch_size, seq_length=128):
    """Load existing compiled model or compile new one for specific batch size"""
    try:
        # Check if compiled model already exists
        filename = f'ner_model_batch_{batch_size}_seq_{seq_length}.pt'
        
        if os.path.exists(filename):
            print(f"\n✅ Found existing compiled model for Batch Size {batch_size}: {filename}")
            print(f"   📁 Skipping compilation, loading existing model...")
            
            # Still need encoder for input preparation
            encoder = RoBERTaNEREncoder()
            
            return {
                'batch_size': batch_size,
                'filename': filename,
                'seq_length': seq_length,
                'encoder': encoder,
                'status': 'loaded'
            }
        
        print(f"\n🔨 No existing model found for Batch Size {batch_size}, compiling new model...")
        
        # Initialize encoder
        encoder = RoBERTaNEREncoder()
        
        # Prepare input for tracing
        input_ids, attention_mask = encoder.encode_for_ner(
            encoder.sample_text, 
            max_length=seq_length, 
            batch_size=batch_size
        )
        
        # Trace the model for Neuron compilation
        logging.info(f"Starting model tracing for batch size {batch_size}...")
        compiler_workdir = f"./compiler_workdir_batch_{batch_size}_seq_{seq_length}"
        model_neuron = torch_neuronx.trace(
            encoder.model, 
            (input_ids, attention_mask),
            compiler_workdir=compiler_workdir
        )
        logging.info(f"Model tracing completed for batch size {batch_size}")
        
        # Save the compiled model
        torch.jit.save(model_neuron, filename)
        logging.info(f"Saved compiled model as {filename}")
        
        return {
            'batch_size': batch_size,
            'filename': filename,
            'seq_length': seq_length,
            'encoder': encoder,
            'status': 'compiled'
        }
        
    except Exception as e:
        print(f"❌ Error processing batch size {batch_size}: {e}")
        import traceback
        traceback.print_exc()
        return None

def benchmark_on_core(model_info, core_id, num_batches=128):
    """Run benchmark for a specific model on a specific NeuronCore"""
    try:
        batch_size = model_info['batch_size']
        filename = model_info['filename']
        seq_length = model_info['seq_length']
        encoder = model_info['encoder']
        
        # Set environment variable to target specific core
        original_core = os.environ.get('NEURON_RT_VISIBLE_CORES', '')
        os.environ['NEURON_RT_VISIBLE_CORES'] = str(core_id)
        
        print(f"\n STARTING: Batch Size {batch_size} on Core {core_id} - Running {num_batches} batches")
        
        # Load the compiled model on this core
        model_neuron = torch.jit.load(filename)
        
        # IMPORTANT: Prepare input with correct batch size for this specific model
        print(f"📦 Preparing inputs with batch_size={batch_size}, seq_length={seq_length}")
        sample_input_ids, sample_attention_mask = encoder.encode_for_ner(
            encoder.sample_text, 
            max_length=seq_length, 
            batch_size=batch_size  # MUST match the compiled model's batch size
        )
        
        print(f" Input shapes: {sample_input_ids.shape}, {sample_attention_mask.shape}")
        
        # Warmup - 8 runs to stabilize performance
        print(" Warming up model (8 runs)...")
        for _ in range(8):
            with torch.no_grad():
                _ = model_neuron(sample_input_ids, sample_attention_mask)
        
        # Benchmark - 128 batches
        print(f"⏱️  Starting benchmark: {num_batches} batches of size {batch_size}")
        latencies = []
        start_time = time.time()
        
        for batch_idx in range(num_batches):
            # Use the same pre-encoded input for consistency (compiled batch size)
            # NOTE: In real usage, you'd encode different texts, but for benchmarking we use same input
            
            # Time individual batch inference
            batch_start = time.time()
            with torch.no_grad():
                logits = model_neuron(sample_input_ids, sample_attention_mask)
            batch_end = time.time()
            
            batch_latency = (batch_end - batch_start) * 1000  # Convert to ms
            latencies.append(batch_latency)
            
            # Progress updates every 10 batches
            if (batch_idx + 1) % 10 == 0:
                progress = (batch_idx + 1) / num_batches * 100
                print(f"    Batch Size {batch_size}: {batch_idx + 1}/{num_batches} batches ({progress:.1f}%) - Latest: {batch_latency:.2f}ms")
        
        end_time = time.time()
        total_duration = end_time - start_time
        
        # Restore original core setting
        if original_core:
            os.environ['NEURON_RT_VISIBLE_CORES'] = original_core
        else:
            os.environ.pop('NEURON_RT_VISIBLE_CORES', None)
        
        # Calculate metrics
        total_inferences = num_batches * batch_size
        throughput = total_inferences / total_duration
        
        latencies.sort()
        p50_latency = latencies[len(latencies) // 2]
        p95_latency = latencies[int(len(latencies) * 0.95)]
        p99_latency = latencies[int(len(latencies) * 0.99)]
        
        # Results
        results = {
            'core_id': core_id,
            'batch_size': batch_size,
            'seq_length': seq_length,
            'num_batches': num_batches,
            'total_inferences': total_inferences,
            'duration_seconds': total_duration,
            'throughput_per_second': throughput,
            'latency_p50_ms': p50_latency,
            'latency_p95_ms': p95_latency,
            'latency_p99_ms': p99_latency,
            'avg_latency_ms': sum(latencies) / len(latencies)
        }
        
        print(f"\n COMPLETED: Batch Size {batch_size} on Core {core_id}")
        print(f"    Total Duration: {total_duration:.3f}s")
        print(f"    Throughput: {throughput:.2f} inferences/sec")
        print(f"    Latency P50: {p50_latency:.3f}ms, P95: {p95_latency:.3f}ms, P99: {p99_latency:.3f}ms")
        print(f"    Model used: {filename}")
        print("   " + "="*80)
        
        return results
        
    except Exception as e:
        print(f"❌ Error benchmarking batch size {batch_size} on core {core_id}: {e}")
        import traceback
        traceback.print_exc()
        return None

def clean_compiled_models(batch_sizes, seq_length=128):
    """Remove existing compiled models to force recompilation"""
    print("🧹 Cleaning existing compiled models...")
    cleaned = 0
    for batch_size in batch_sizes:
        filename = f'ner_model_batch_{batch_size}_seq_{seq_length}.pt'
        if os.path.exists(filename):
            os.remove(filename)
            print(f"   🗑️  Removed: {filename}")
            cleaned += 1
    print(f"✅ Cleaned {cleaned} existing models")

def get_available_cores():
    """Get number of available NeuronCores"""
    try:
        # Check if NEURON_RT_NUM_CORES is set
        num_cores = int(os.environ.get('NEURON_RT_NUM_CORES', '1'))
        return list(range(num_cores))
    except:
        # Default to single core if detection fails
        return [0]

def main():
    """Load/compile models, then run benchmarks in parallel or sequential"""
    try:
        print("=== Parallel Multi-Core NER Benchmark with AWS Neuron ===")
        print("🚀 Smart model loading: Will use existing compiled models if available")
        
        # Configuration
        batch_sizes = [4, 8, 16, 32]  # Adjust based on your cores
        seq_length = 128
        num_batches = 128
        force_recompile = False  # Set to True to force recompilation of all models
        
        # Optional: Clean existing models to force recompilation
        if force_recompile:
            clean_compiled_models(batch_sizes, seq_length)
        
        # Step 1: Load existing or compile new models
        print("\n" + "="*60)
        print("PHASE 1: LOADING OR COMPILING MODELS")
        print("="*60)
        
        compiled_models = []
        loaded_count = 0
        compiled_count = 0
        
        for batch_size in batch_sizes:
            model_info = load_or_compile_model_for_batch_size(batch_size, seq_length)
            if model_info:
                compiled_models.append(model_info)
                if model_info['status'] == 'loaded':
                    loaded_count += 1
                else:
                    compiled_count += 1
        
        if not compiled_models:
            print("❌ No models available!")
            return
        
        print(f"✅ Ready to benchmark {len(compiled_models)} models:")
        print(f"   📁 Loaded existing: {loaded_count}")
        print(f"   🔨 Newly compiled: {compiled_count}")
        print(f"   📋 Batch sizes: {[m['batch_size'] for m in compiled_models]}")
        
        # Step 2: Get available cores
        available_cores = get_available_cores()
        print(f"🖥️  Detected {len(available_cores)} NeuronCores: {available_cores}")
        
        # Step 3: Run benchmarks (parallel if multiple cores, sequential if single core)
        print("\n" + "="*60)
        print("PHASE 2: BENCHMARKING ALL BATCH SIZES")
        print("="*60)
        
        all_results = []
        
        if len(available_cores) >= len(compiled_models):
            # Parallel execution - each model on different core
            print("Running in PARALLEL mode (each batch size on different core)")
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(compiled_models)) as executor:
                future_to_info = {}
                
                for i, model_info in enumerate(compiled_models):
                    core_id = available_cores[i % len(available_cores)]
                    future = executor.submit(benchmark_on_core, model_info, core_id, num_batches)
                    future_to_info[future] = (model_info['batch_size'], core_id)
        else:
            # Sequential execution on single core - but run ALL models
            print(f"Running in SEQUENTIAL mode on Core 0 (only {len(available_cores)} core available)")
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future_to_info = {}
                
                for model_info in compiled_models:  # Run ALL compiled models
                    core_id = 0  # Use core 0 for all
                    future = executor.submit(benchmark_on_core, model_info, core_id, num_batches)
                    future_to_info[future] = (model_info['batch_size'], core_id)
        
        # Collect results as they complete
        for future in concurrent.futures.as_completed(future_to_info):
            batch_size, core_id = future_to_info[future]
            try:
                result = future.result()
                if result:
                    all_results.append(result)
                    print(f"✅ FINISHED: Batch Size {batch_size} on Core {core_id} - Added to results")
            except Exception as exc:
                print(f"❌ FAILED: Batch Size {batch_size} on Core {core_id} - Error: {exc}")
        
        # Step 4: Save and display results
        if all_results:
            # Save results to JSON
            with open('parallel_benchmark_results.json', 'w') as f:
                json.dump(all_results, f, indent=2)
            
            # Print summary table
            print("\n" + "="*120)
            print(f"BENCHMARK SUMMARY - {len(all_results)} MODELS TESTED")
            print("="*120)
            print(f"{'Core':<6} {'Batch Size':<12} {'Duration(s)':<12} {'Throughput':<15} {'P50 Lat(ms)':<12} {'P95 Lat(ms)':<12} {'P99 Lat(ms)':<12}")
            print("-"*120)
            
            # Sort by core_id then batch_size
            all_results.sort(key=lambda x: (x['core_id'], x['batch_size']))
            
            for result in all_results:
                print(f"{result['core_id']:<6} {result['batch_size']:<12} {result['duration_seconds']:<12.3f} "
                      f"{result['throughput_per_second']:<15.2f} {result['latency_p50_ms']:<12.3f} "
                      f"{result['latency_p95_ms']:<12.3f} {result['latency_p99_ms']:<12.3f}")
            
            print(f"\n🎉 SUCCESS: Benchmarked {len(all_results)} batch sizes!")
            print(f"   📊 Total batches executed: {sum(r['num_batches'] for r in all_results)}")
            print(f"   📁 Loaded existing models: {loaded_count}")
            print(f"   🔨 Newly compiled models: {compiled_count}")
            print("   📁 Results saved to: parallel_benchmark_results.json")
            print("="*120)
        else:
            print("❌ No benchmark results collected!")
        
    except Exception as e:
        print(f"❌ Error in main(): {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()