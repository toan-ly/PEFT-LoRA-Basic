import argparse
from model import load_lora_model
from data import load_and_format_dataset
from trl import SFTTrainer, TrainingArguments

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="unsloth/llama-3-1b-bnb-4bit")
    parser.add_argument("--train_file", type=str, default="train.json")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--max_steps", type=int, default=200)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--output_dir", type=str, default="checkpoints")
    args = parser.parse_args()

    model, tokenizer = load_lora_model(args.model_name)

    dataset = load_and_format_dataset(args.train_file)

    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset,
        dataset_text_field = "text",
        max_seq_length = 512,
        dataset_num_proc = 4,
        packing = False,
        args = TrainingArguments(
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            warmup_steps=5,
            max_steps=args.max_steps,
            logging_steps=5,
            save_steps=50,
            output_dir=args.output_dir,
            optim="paged_adamw_8bit",
            lr_scheduler_type="cosine",
            learning_rate=args.lr,
            fp16=True,
            bf16=False,
            save_total_limit=2,
            report_to="none",
        ),
    )
    trainer.train()
    model.save_pretrained(f"{args.output_dir}/final")

if __name__ == "__main__":
    main()