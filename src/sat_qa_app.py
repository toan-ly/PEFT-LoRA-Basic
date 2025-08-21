import gradio as gr
import os
import base64
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline
)

from template import generate_prompt_template
from dotenv import load_dotenv

load_dotenv()

# from huggingface_hub import login
# login(token="your_token")

MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    device_map="auto",
)

generator = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device_map="auto",
    framework='pt'
)


def query_llm_for_sat(passage, question, choices):
    if not passage.strip() or not question.strip():
        return "Please provide both passage and question."

    choices_dict = {}
    for line in choices.strip().split('\n'):
        if line and ':' in line:
            k, v = line.split(':', 1)
            choices_dict[k.strip()] = v.strip()
        
    if not all(k in choices_dict for k in ['A', 'B', 'C', 'D']):
        return "Please provide all choices (A, B, C, D)."

    prompt = generate_prompt_template(passage, question, choices_dict)

    try:
        outputs = generator(prompt, max_new_tokens=256, do_sample=False, pad_token_id=tokenizer.eos_token_id)
        full = outputs[0]['generated_text']
        return full[len(prompt):].strip()
    except Exception as e:
        return "Error during inference: {e}"

def load_example():
    passage_text = (
        "This passage is adapted from F. Scott Fitzgerald, The Great Gatsby.\n"
        '"In my younger and more vulnerable years my father gave me some advice '
        "that I’ve been turning over in my mind ever since. ‘Whenever you feel "
        "like criticizing anyone,’ he told me, ‘just remember that all the people "
        "in this world haven’t had the advantages that you’ve had.’ He didn’t say "
        "any more, but we’ve always been unusually communicative in a reserved way, "
        "and I understood that he meant a great deal more than that. In consequence, "
        "I’m inclined to reserve all judgments, a habit that has opened up many "
        'curious natures to me and also made me the victim of not a few veteran bores."'
    )
    question_text = "What is the primary purpose of the narrator’s recollection of his father’s advice?"
    choices_text = "\n".join([
        "A: To explain his reluctance to judge others",
        "B: To highlight his privileged upbringing",
        "C: To criticize his father’s moral values",
        "D: To foreshadow future conflicts in the story"
    ])
    note_text = "**💡 Note:** For this example, **A** is the correct answer."
    return passage_text, question_text, choices_text, note_text

def img_to_base64(img_path):
    try:
        with open(img_path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except Exception as e:
        return ""

def create_header():
    with gr.Column(scale=4):
        gr.Markdown(
            """
            <div style="display: flex; justify-content: space-between; align-items: center; padding: 0 15px;">
                <div>
                    <h1 style="margin-bottom: 0;">📚 SAT Reading Question Assistant</h1>
                    <p style="margin-top: 0.5em; color: #666;">🧠 Using LLM to solve SAT Reading Questions</p>
                    <p style="margin-top: 0.2em; color: #7f8c8d;">🎯 Input a passage, question, and answer choices to get the correct answer</p>
                </div>
            </div>
            """)

def build_interface():
    css = """
    .gradio-container{min-height:100vh;}
    .btn {
        width: 100%;
        height: 45px;
        font-size: 16px;
        margin-bottom: 10px;
        background: linear-gradient(45deg,#FF6B6B,#4ECDC4);
        color: #fff;
        border: none;
        border-radius: 5px;
    }
    .btn:hover {
        background: linear-gradient(45deg,#FF5252,#3CB4AC);
    }
    .note {
        padding: 10px;
        background: #e8f6ff;
        border-left: 4px solid #3498db;
        border-radius: 4px;
        font-style: italic;
        color: #2c3e50;
        margin-bottom: 10px;
    }
    """
    with gr.Blocks(css=css) as app:
        create_header()
        with gr.Row():
            # Left col: Inputs
            with gr.Column(scale=1):
                passage = gr.Textbox(placeholder="Passage...", label="Passage", lines=12)
                question = gr.Textbox(placeholder="Question...", label="Question", lines=3)
                choices = gr.Textbox(value="A:\nB:\nC:\nD:\n", label="Choices", lines=5)   
            
            # Right col: Example, Answer
            with gr.Column(scale=1):
                # Buttons
                example_btn = gr.Button("Load Example 📖 (Answer: A)", elem_classes="btn")
                process_btn = gr.Button("Get Answer 📚", elem_classes="btn")

                answer = gr.Textbox(label="Answer", lines=15)

                example_btn.click(fn=load_example, outputs=[passage, question, choices])
                process_btn.click(fn=query_llm_for_sat, inputs=[passage, question, choices], outputs=answer)

    return app



if __name__ == "__main__":
    app = build_interface()
    app.launch(server_name="0.0.0.0", server_port=7860)