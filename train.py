import os
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer

class JESTDataSelector:
    """
    Mock implementation of JEST (Joint Example Selection for Training)
    In a real scenario, this would use a reference model to score learnability of batches.
    Here we recommend high-quality datasets based on the user's target domain.
    """
    def __init__(self):
        self.datasets = {
            "general": "wikitext",
            "chat": "HuggingFaceH4/ultrachat_200k",
            "code": "bigcode/starcoderdata",
            "math": "MetaMathQA"
        }

    def recommend(self, domain):
        domain = domain.lower()
        if domain in self.datasets:
            print(f"[JEST Data Selector] Recommended dataset for '{domain}': {self.datasets[domain]}")
            return self.datasets[domain]
        else:
            print("[JEST Data Selector] Domain not recognized. Defaulting to general data: wikitext")
            return self.datasets["general"]


def main():
    print("=== Local LLM Training (Fine-tuning) with JEST ===")

    # 1. JEST Data Selection
    print("\n--- Data Selection (JEST) ---")
    print("Available domains for recommendation: general, chat, code, math")
    domain = input("Enter the domain for recommendation, or type 'custom' to provide a link: ").strip()

    dataset_name = None
    if domain.lower() == 'custom':
        dataset_name = input("Enter the Hugging Face dataset link or ID (e.g., 'tatsu-lab/alpaca'): ").strip()
    else:
        jest = JESTDataSelector()
        dataset_name = jest.recommend(domain)

    print(f"\nProceeding with dataset: {dataset_name}")
    try:
        # Load a small sample of the dataset for quick local training simulation
        if dataset_name == "wikitext":
            dataset = load_dataset(dataset_name, "wikitext-2-raw-v1", split="train[:100]")
        else:
            dataset = load_dataset(dataset_name, split="train[:100]")
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        print("Falling back to a small default dataset 'tatsu-lab/alpaca'.")
        dataset_name = "tatsu-lab/alpaca"
        dataset = load_dataset(dataset_name, split="train[:100]")

    print("Dataset loaded successfully.\n")

    # 2. Configure Model and LoRA
    model_id = "Qwen/Qwen1.5-0.5B-Chat"
    print(f"--- Loading Base Model ({model_id}) ---")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None
    )

    print("\n--- Configuring LoRA Adapters ---")
    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj"]
    )
    # 3. Training Setup
    output_dir = "./model_adapters"
    print(f"\n--- Starting Training ---")
    print(f"Adapters will be saved to: {output_dir}")

    dataset_text_field = "text"
    if "text" not in dataset.column_names:
        dataset_text_field = dataset.column_names[0] # Try to guess

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=2,
        max_steps=1, # Very small number for quick simulation
        logging_steps=2,
        learning_rate=2e-4,
        use_cpu=True
    )

    # SFTTrainer wrapper for quick prototyping without processing class errors in old versions
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=training_args,
        peft_config=peft_config,
        dataset_text_field=dataset_text_field
    )

    trainer.train()

    # 4. Save the trained parameters
    trainer.model.save_pretrained(output_dir)
    print(f"\nTraining complete. Parameter files saved to {output_dir}.")

if __name__ == "__main__":
    main()
