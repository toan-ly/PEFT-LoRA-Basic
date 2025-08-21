def infer(model_path="thainq107/med-mcqa-llama-3.2-1B-4bit-lora", prompt="Question: What is the capital of France?\nChoices:\nA. Berlin\nB. Paris\nC. Madrid\nD. Rome\nAnswer:"):
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_path,
        max_seq_length = 2048,
        dtype = "auto",
        load_in_4bit = True,
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=10)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(answer)

if __name__ == "__main__":
    infer()