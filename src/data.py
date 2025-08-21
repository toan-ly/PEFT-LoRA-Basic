from datasets import load_dataset

data_prompt = """Choose the correct option for the following question.

### Question:
{}

### Choice:
{}

### Answer:
"""

id2label = {
    0: 'A',
    1: 'B',
    2: 'C',
    3: 'D'
}


def formatting_prompt(examples):
    questions = examples["question"]
    opas = examples["opa"]
    opbs = examples["opb"]
    opcs = examples["opc"]
    opds = examples["opd"]
    cops = examples["cop"]

    texts = []
    for idx in range(len(questions)):
        question = questions[idx]
        opa = opas[idx]
        opb = opbs[idx]
        opc = opcs[idx]
        opd = opds[idx]
        # answer = id2label[cops[idx]]
        # if answer == "A":
        #     answer = answer + " " + opa
        # elif answer == "B":
        #     answer = answer + " " + opb
        # elif answer == "C":
        #     answer = answer + " " + opc
        # elif answer == "D":
        #     answer = answer + " " + opd

        choices = f"A. {opa}. B. {opb}. C. {opc}. D. {opd}."
        text = data_prompt.format(question, choices)
        texts.append(text)
    return {"text": texts,}


def load_and_format_dataset(path="openlifescienceai/medmcqa", split="train"):
    dataset = load_dataset(path)
    dataset = dataset.map(formatting_prompt)
    del dataset["test"]
    return dataset