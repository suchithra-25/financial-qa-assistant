import streamlit as st
import pandas as pd
import pdfplumber
import ollama
import re

# --- Helper functions ---

def extract_text_from_pdf(pdf_file):
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text

def extract_text_from_excel(excel_file):
    df = pd.read_excel(excel_file, sheet_name=None)  # read all sheets
    text = ""
    for sheet_name, sheet_data in df.items():
        text += f"\n--- Sheet: {sheet_name} ---\n"
        text += sheet_data.to_string(index=False)
    return text

def chunk_text(text, chunk_size=700):
    # Split into smaller parts for faster + accurate responses
    words = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

def search_relevant_chunks(chunks, query, top_n=3):
    # Simple keyword-based relevance (fast, no external libs)
    scored = []
    for chunk in chunks:
        score = sum(query.lower().count(word.lower()) for word in chunk.split())
        scored.append((score, chunk))
    scored.sort(reverse=True, key=lambda x: x[0])
    return [c for _, c in scored[:top_n]]

def ask_ollama(prompt, model="gemma:2b"):
    response = ollama.chat(model=model, messages=[{"role": "user", "content": prompt}])
    return response['message']['content']


# --- Streamlit UI ---

st.set_page_config(page_title="Financial Q&A Assistant", layout="wide")
st.title("📊 Financial Document Q&A Assistant")

uploaded_file = st.file_uploader("Upload a PDF or Excel file", type=["pdf", "xlsx", "xls"])

if uploaded_file:
    st.success("✅ File uploaded successfully!")

    # Extract + chunk text
    if uploaded_file.name.endswith(".pdf"):
        raw_text = extract_text_from_pdf(uploaded_file)
    else:
        raw_text = extract_text_from_excel(uploaded_file)

    chunks = chunk_text(raw_text)
    st.info(f"Document processed into {len(chunks)} chunks for faster Q&A.")

    # Chat interface
    user_question = st.text_input("Ask a question about your financial document:")

    if user_question:
        with st.spinner("Thinking..."):
            relevant_chunks = search_relevant_chunks(chunks, user_question)
            context = "\n\n".join(relevant_chunks)

            prompt = f"""
            You are a financial assistant. 
            Use the following financial data to answer the question.

            Context:
            {context}

            Question: {user_question}
            """

            answer = ask_ollama(prompt, model="gemma:2b")

        st.subheader("Answer")
        st.write(answer)







