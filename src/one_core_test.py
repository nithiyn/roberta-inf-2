import logging
import torch
import torch_neuronx

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    datefmt='%H:%M:%S'
)
from bert_encode import BERTEncoder

def main():
    try:
        print("=== Method 1: Basic Usage ===")
        encoder = BERTEncoder()
        logging.info(f"Traced model...")
        paraphrase = encoder.encode(encoder.sequence_0, encoder.sequence_2)
        not_paraphrase = encoder.encode(encoder.sequence_0, encoder.sequence_1)
        logging.info(f"Traced model...")
        # Compile the model for Neuron
        logging.info(f"Tracing model...")
        model_neuron = torch_neuronx.trace(encoder.model, paraphrase)
        logging.info(f"Traced model...")
        # Save the TorchScript for inference deployment
        filename = 'model_1024.pt'
        torch.jit.save(model_neuron, filename)
        # Load the TorchScript compiled model
        model_neuron = torch.jit.load(filename)

        # Verify the TorchScript works on both example inputs
        logging.info(f"logits")
        neuron_paraphrase_logits = model_neuron(*paraphrase)[0]
        neuron_not_paraphrase_logits = model_neuron(*not_paraphrase)[0]
        logging.info(f"LOGITS")

        print('Neuron paraphrase logits:    ', neuron_paraphrase_logits.detach().numpy())
        print('Neuron not-paraphrase logits: ', neuron_not_paraphrase_logits.detach().numpy())

    except Exception as e:
        print(f"❌ Error in main(): {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()  # ← This line is missing!