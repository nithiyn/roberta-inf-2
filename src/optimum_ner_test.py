import os
from optimum.neuron import AutoModelForTokenClassification
from transformers import AutoTokenizer, pipeline

def main():
    """Alternative NER implementation using Optimum Neuron"""
    try:
        print("=== NER with Optimum Neuron ===")
        
        # 1. Define Model and Core Parameters
        # Using a popular NER model from the Hugging Face Hub
        model_id = "Davlan/xlm-roberta-base-ner-hrl"
        # Directory to save the compiled Neuron model
        save_directory = "neuron_ner_model"
        
        print(f"Loading model '{model_id}' for compilation to AWS Neuron format...")
        
        # 2. Compile and Save the Model for Neuron
        # The `export=True` argument triggers the compilation from PyTorch to Neuron.
        # We define the expected input shapes for optimization.
        compiler_args = {
            "export": True, 
            "input_shapes": {"batch_size": 1, "sequence_length": 128}
        }
        
        # Load the tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        
        # Load and compile the model
        neuron_model = AutoModelForTokenClassification.from_pretrained(
            model_id,
            **compiler_args,
        )
        
        # Save the compiled model and tokenizer to a local directory for reuse
        neuron_model.save_pretrained(save_directory)
        tokenizer.save_pretrained(save_directory)
        
        print(f"Model successfully compiled and saved to '{save_directory}'")
        print("-" * 50)
        
        # 3. Load the Compiled Model and Run Inference
        print("Loading the compiled Neuron model for inference...")
        
        # Load the pre-compiled model from the directory
        compiled_model = AutoModelForTokenClassification.from_pretrained(save_directory)
        compiled_tokenizer = AutoTokenizer.from_pretrained(save_directory)
        
        # Create a token-classification pipeline using the Neuron-optimized model
        ner_pipeline = pipeline(
            "token-classification",
            model=compiled_model,
            tokenizer=compiled_tokenizer,
        )
        
        # 4. Perform NER on example sentences
        test_sentences = [
            "My name is Clara and I live in San Francisco with my friend Wolfgang.",
            "The company HuggingFace is based in New York City.",
            "Apple Inc. was founded by Steve Jobs in California."
        ]
        
        for i, text in enumerate(test_sentences, 1):
            print(f"\n--- Test {i}: Running NER on sentence: '{text}' ---")
            
            ner_results = ner_pipeline(text)
            
            # Print the results in a readable format
            print("NER Results:")
            if ner_results:
                for entity in ner_results:
                    # '##' tokens are subwords; let's merge them for cleaner output
                    if entity['word'].startswith('##'):
                        continue
                    print(
                        f"  - Entity: {entity['word']:<15} | "
                        f"Type: {entity['entity']:<10} | "
                        f"Score: {entity['score']:.4f}"
                    )
            else:
                print("  No entities found")
        
        print("\n=== Optimum Neuron NER completed successfully! ===")
        
    except Exception as e:
        print(f"❌ Error in optimum neuron test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()