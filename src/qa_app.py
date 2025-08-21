import streamlit as st
from transformers import pipeline


generator = pipeline("text-generation", model="thainq107/med-mcqa-llama-3.2-1B-4bit-lora")

def main():
    st.title("Multiple Choice Question Answering")
    st.subheader('Model: LLaMA-3.2-1B. Dataset: MedMCQA')

    question = """"Choose the correct option for the following question.\n
    ### Question:\nWhich of the following is not true for myelinated nerve fibers:\n
    ### Choice:\n
    A. Impulse through myelinated fibers is slower than non-myelinated fibers. 
    B. Membrane currents are generated at nodes of Ranvier. 
    C. Saltatory conduction of impulses is seen. 
    D. Local anesthesia is effective only when the nerve is not covered by myelin sheath.\n\n
    ### Answer:\n
    """
    text_input = st.text_input("Prompt: ", question)
    response = generator(text_input, max_length=1024, return_full_text=False)
    st.write(text_input)
    st.write(response['generated_text'])

if __name__ == "__main__":
    main()
