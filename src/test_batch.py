# CORRECTED VERSION - batch_processing.py

import logging
import torch
import torch_neuronx
from benchmark import benchmark
from bert_encode import BERTEncoder

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    datefmt='%H:%M:%S'
)

def main():
    logging.info("🚀 Starting BERT batch processing...")
    
    # Initialize encoder ONCE (more efficient)
    logging.info("📥 Initializing BERTEncoder...")
    encoder = BERTEncoder(model_name="bert-base-cased-finetuned-mrpc")  # Specify model name
    logging.info("✅ Encoder initialized")
    
    # Compile BERT for different batch sizes
    logging.info("🔧 Starting model compilation for different batch sizes...")
    
    try:
        for batch_size in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
            logging.info(f"⚡ Compiling model for batch size {batch_size}...")
            
            # Use your encoder class properly
            example = encoder.encode(encoder.sequence_0, encoder.sequence_2, batch_size=batch_size)
            
            # Trace using the encoder's model
            model_neuron = torch_neuronx.trace(encoder.model, example)
            
            # Save the traced model
            filename = f'model_batch_size_{batch_size}.pt'
            torch.jit.save(model_neuron, filename)
            logging.info(f"✅ Saved {filename}")
            
    except Exception as e:
        logging.error(f"❌ Compilation error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
    
    logging.info("🎯 Model compilation completed!")
    
    # Benchmark BERT for different batch sizes
    logging.info("📊 Starting benchmarking...")
    
    try:
        for batch_size in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
            print('-'*60)
            logging.info(f"🏃 Benchmarking batch size {batch_size}...")
            
            # Create example for this batch size
            example = encoder.encode(encoder.sequence_0, encoder.sequence_2, batch_size=batch_size)
            filename = f'model_batch_size_{batch_size}.pt'
            
            # Run benchmark
            metrics = benchmark(filename, example)
            
            # Log key metrics
            logging.info(f"✅ Batch {batch_size}: {metrics['throughput']:.2f} inferences/sec")
            print()
            
    except Exception as e:
        logging.error(f"❌ Benchmark execution failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
    
    logging.info("🎉 All benchmarking completed successfully!")

if __name__ == "__main__":
    main()