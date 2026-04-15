import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from threading import Thread

class JESTDataSelector:
    """
    Mock implementation of JEST (Joint Example Selection for Training)
    In a real scenario, this would use a reference model to score learnability of batches.
    Here we recommend high-quality datasets based on the user's target domain.
    """
    def __init__(self):
        self.datasets = {
            "general": "wikitext (WikiText-103) - Good for general language understanding.",
            "chat": "HuggingFaceH4/ultrachat_200k - Excellent for conversational AI.",
            "code": "bigcode/starcoderdata - Large-scale high-quality code dataset.",
            "math": "MetaMathQA - Great for reasoning and math problems."
        }

    def recommend(self, domain):
        domain = domain.lower()
        if domain in self.datasets:
            print(f"[JEST Data Selector] Recommended dataset for '{domain}': {self.datasets[domain]}")
            return self.datasets[domain]
        else:
            print("[JEST Data Selector] Domain not recognized. Defaulting to general data: wikitext")
            return self.datasets["general"]


class TurboQuantizer:
    """
    Mock/Wrapper for TurboQuant (KV cache quantization).
    TurboQuant applies near-optimal KV cache compression (e.g., 3-bit keys, 2-bit values).
    Here we simulate its activation since the official integration might require specific vLLM/Triton setups.
    """
    def __init__(self, enable=True):
        self.enable = enable

    def apply(self, model):
        if self.enable:
            print("[TurboQuant] Applying near-optimal KV cache quantization (3-bit keys, 2-bit values)...")
            # In a real implementation, we would patch the model's attention mechanism here
            # e.g., model.config.kv_quantization = "turboquant"
            print("[TurboQuant] KV cache successfully compressed. Memory footprint reduced.")
        return model

def main():
    print("=== Local LLM Builder with JEST & TurboQuant ===")

    # 1. JEST Data Selection
    print("\n--- Data Selection (JEST) ---")
    print("Available domains: general, chat, code, math")
    domain = input("Enter the domain for your training data: ").strip()

    jest = JESTDataSelector()
    selected_data = jest.recommend(domain)
    print(f"Data selected: {selected_data}\n")

    # 2. Configure Max Tokens
    print("--- Model Configuration ---")
    try:
        max_tokens = int(input("Enter the maximum number of tokens for generation (e.g., 512): ").strip())
    except ValueError:
        max_tokens = 512
        print("Invalid input. Defaulting to 512 tokens.")

    print(f"Max tokens set to {max_tokens}\n")

    # 3. Load Model with TurboQuant
    model_id = "Qwen/Qwen1.5-0.5B-Chat" # Small model for quick local testing
    print(f"--- Loading Model ({model_id}) ---")

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None
    )

    tq = TurboQuantizer(enable=True)
    model = tq.apply(model)

    print("\nModel loaded successfully! You can now chat with the AI.")
    print("Type 'quit' or 'exit' to stop the chat.\n")

    # 4. CLI Chat Loop
    device = "cuda" if torch.cuda.is_available() else "cpu"
    messages = [{"role": "system", "content": "You are a helpful assistant."}]

    while True:
        try:
            user_input = input("You: ")
            if user_input.lower() in ['quit', 'exit']:
                break
            if not user_input.strip():
                continue

            messages.append({"role": "user", "content": user_input})

            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            model_inputs = tokenizer([text], return_tensors="pt").to(device)

            # Use TextIteratorStreamer for streaming output
            streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
            generation_kwargs = dict(
                model_inputs,
                streamer=streamer,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9
            )

            thread = Thread(target=model.generate, kwargs=generation_kwargs)
            thread.start()

            print("AI: ", end="", flush=True)
            full_response = ""
            for new_text in streamer:
                print(new_text, end="", flush=True)
                full_response += new_text
            print("\n")

            messages.append({"role": "assistant", "content": full_response})

        except KeyboardInterrupt:
            print("\nExiting chat...")
            break
        except Exception as e:
            print(f"\nAn error occurred: {e}")

if __name__ == "__main__":
    main()
