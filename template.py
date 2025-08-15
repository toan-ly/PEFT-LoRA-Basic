def generate_prompt_template(passage, question, choices):
    prompt = f"""
**Task:** Read the passage and answer the question.

---

**Passage:**  
{passage}

---

**Question:**  
{question}

**Choices:**  
A) {choices['A']}  
B) {choices['B']}  
C) {choices['C']}  
D) {choices['D']}

---

**Respond with ONLY the letter and full text of the correct answer.**  

"""
    return prompt