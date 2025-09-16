# financial-qa-assistant
A Streamlit-based Q&amp;A Assistant for financial Excel/PDF files using Ollama (Mistral/Gemma).


# 📊 Financial Document Q&A Assistant

This project is a **Q&A Assistant for financial documents** (Excel or PDF).  
It lets you upload financial records and ask natural language questions like:  
- "What is the total income in January?"  
- "How much was spent on groceries?"  
- "List all expenses above 10,000."  


## 🚀 Features
- Upload **Excel (XLSX/XLS)** or **PDF** files.  
- Ask questions in natural language.  
- Uses **Ollama** with LLMs (Mistral, Gemma) to answer.  
- Works with small, medium, and large datasets (but speed varies). 


## Files
- `app.py` - Streamlit application
- `utils.py` - Extraction helpers and Ollama integration
- `requirements.txt` - Python dependencies


## Models Used

**Mistral (ollama pull mistral)**

-Good accuracy.
-Works fine for small datasets.
-For medium datasets (~200 rows) → takes 1–2 minutes to respond.
-For large datasets (thousands of rows / big PDFs) → can take ~10 minutes.

**Gemma (ollama pull gemma:2b)**

-Lightweight, faster than Mistral.
-But works best only on small files (quick answers).
-Struggles with larger datasets (still slow).


## Performance Notes

-On my system: Intel Iris Xe Graphics (integrated GPU).
-Ollama does not support CUDA Toolkit here (CUDA only works on NVIDIA GPUs).
-That means Ollama runs on CPU only, which makes responses slower.

-If you want 10x faster answers, you need a machine with an **NVIDIA GPU + CUDA Toolkit**.


## Example Questions

-What is the total expense in February?
-Which category has the highest spending?
-Show me all income transactions.
-Average monthly rent expense?
