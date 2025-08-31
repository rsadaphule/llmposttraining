import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

MODEL_PATH = "rsadaphule/qwen-alpaca-finetuned"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)


# Use GPU if available
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

st.set_page_config(page_title="Reasoning AI Assistant", layout="centered")
st.title("🤖 Reasoning & Planning AI Assistant")
st.write("Ask a complex question or give an instruction, and see how the AI responds with reasoning and planning.")

# User input
user_query = st.text_area("Your Question/Instruction:", 
                          placeholder="e.g., Plan a healthy meal for my lunch break with ingredients: chicken, avocado, lettuce.")
submit_button = st.button("Get Answer")

if submit_button:
    if user_query.strip() == "":
        st.warning("Please enter a question or instruction.")
    else:
        # Display a placeholder while processing
        with st.spinner("Thinking..."):
            # Encode the input and generate response
            inputs = tokenizer(user_query, return_tensors="pt").to(device)
            outputs = model.generate(**inputs, 
                                     max_new_tokens=200, 
                                     pad_token_id=tokenizer.eos_token_id,
                                     eos_token_id=tokenizer.eos_token_id)
            answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
        st.write("**Assistant:** " + answer)