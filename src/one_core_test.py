print("🚀 Script is starting...")

import os
import logging
import torch
import torch_neuronx
from bert_encode import RoBERTaNEREncoder

# Set up environment variables for Neuron compilation
os.environ['NEURON_CC_FLAGS'] = "--model-type=transformer --dump=../ncc_dump --cache_dir=./neuron_cache"
os.environ["NEURON_FRAMEWORK_DEBUG"] = "1"

# Create directories
os.makedirs("../ncc_dump", exist_ok=True)
os.makedirs("./compiler_workdir", exist_ok=True)


# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', datefmt='%H:%M:%S')

def main():
    try:
        print("=== NER with AWS Neuron - Single Core Test ===")

        # Initialize the NER encoder
        encoder = RoBERTaNEREncoder()
        logging.info("Loaded NER model and tokenizer")

        # Prepare sample inputs
        sample_texts = [
            "The company HuggingFace is based in New York City",
            "Apple Inc. was founded by Steve Jobs in California", 
            "Microsoft is headquartered in Redmond, Washington"
        ]

        # Encode the first sample for tracing
        batch_size = 8 # Default batch size
        input_ids, attention_mask = encoder.encode_for_ner(sample_texts[0], batch_size=batch_size)
        logging.info("Encoded sample text for tracing")

        # Trace the model for Neuron compilation
        logging.info("Starting model tracing...")
        compiler_workdir = f"./compiler_workdir_singlebatch{batch_size}"
        model_neuron = torch_neuronx.trace(
            encoder.model, 
            (input_ids, attention_mask),
            compiler_workdir=compiler_workdir
        )
        logging.info("Model tracing completed")

        # Save the compiled model
        filename = f'ner_model_robertabatch{batch_size}.pt'
        torch.jit.save(model_neuron, filename)
        logging.info(f"Saved compiled model as {filename}")

        # Load and test the compiled model
        model_neuron = torch.jit.load(filename)

        # Test on all sample texts
        for i, text in enumerate(sample_texts):
            print(f"\n--- Testing text {i+1}: {text} ---")

            # Encode the text
            test_input_ids, test_attention_mask = encoder.encode_for_ner(text, batch_size=batch_size)

            # Run inference with compiled model
            with torch.no_grad():
                logits = model_neuron(test_input_ids, test_attention_mask)

            # Decode predictions
            entities = encoder.decode_ner_predictions(logits, test_input_ids)

            # Print results
            if entities[0]:  # Check first batch item
                print("Found entities:")
                for entity in entities[0]:
                    print(f"  {entity['token']} -> {entity['label']} (confidence: {entity['confidence']:.3f})")
            else:
                print("No entities found")

        print("\n=== NER with Neuron completed successfully! ===")

    except Exception as e:
        print(f"❌ Error in main(): {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()