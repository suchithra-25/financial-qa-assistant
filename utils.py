import io
import re
import subprocess
from typing import Tuple, Dict, Any

import pandas as pd
import pdfplumber


def extract_from_pdf(file_bytes: bytes) -> Dict[str, Any]:
    """
    Extract text and tables from a PDF file given as bytes.
    Returns a dict with keys: 'text' (str) and 'tables' (list of DataFrames).
    """
    text_parts = []
    tables = []

    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        for page in pdf.pages:
            # text
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)

            # tables: page.extract_tables returns list of list of rows (may be None)
            try:
                page_tables = page.extract_tables()
            except Exception:
                page_tables = None

            if page_tables:
                for t in page_tables:
                    # Convert to DataFrame; first row often headers (if not, result will still be readable)
                    try:
                        df = pd.DataFrame(t[1:], columns=t[0])
                    except Exception:
                        df = pd.DataFrame(t)
                    tables.append(df)

    return {"text": "\n\n".join(text_parts), "tables": tables}


def extract_from_excel(file_bytes: bytes) -> Dict[str, Any]:
    """
    Read Excel file bytes. Returns a dict with keys: 'text' (str) and 'tables' (list of DataFrames).
    'text' will be a plain-text serialization of sheets; 'tables' contains DataFrames for each sheet.
    """
    tables = []
    text_parts = []

    # pandas can read from BytesIO
    xls = pd.ExcelFile(io.BytesIO(file_bytes))
    for sheet_name in xls.sheet_names:
        try:
            df = xls.parse(sheet_name=sheet_name, header=0, dtype=str)
        except Exception:
            # try without header
            df = xls.parse(sheet_name=sheet_name, header=None, dtype=str)
        tables.append(df)
        text_parts.append(f"Sheet: {sheet_name}\n{df.to_string(index=False)}")

    return {"text": "\n\n".join(text_parts), "tables": tables}


def summarize_tables(tables):
    """
    Create a short, readable summary of extracted tables (first few rows and inferred numeric columns).
    Returns a string summary suitable to include in the LLM context.
    """
    parts = []
    for i, df in enumerate(tables):
        parts.append(f"--- Table {i + 1} ---")
        # show up to first 8 rows
        try:
            parts.append(df.head(8).to_string(index=False))
        except Exception:
            parts.append(str(df))
        # try to find numeric columns and last-year values heuristically
        numeric_cols = []
        for col in df.columns:
            # attempt to coerce values to numeric ignoring commas and currency signs
            try:
                series = pd.to_numeric(df[col].astype(str).str.replace(r"[^\d\.\-]", "", regex=True), errors="coerce")
                if series.notna().sum() > 0:
                    numeric_cols.append(col)
            except Exception:
                continue
        if numeric_cols:
            parts.append("Numeric columns detected: " + ", ".join(numeric_cols))
    return "\n\n".join(parts)


def build_context(extracted: Dict[str, Any], max_chars: int = 35000) -> str:
    """
    Build a single string context from extracted items (text + table summaries).
    Trim if too long.
    """
    text = extracted.get("text", "") or ""
    tables = extracted.get("tables", []) or []
    table_summary = summarize_tables(tables) if tables else ""
    context = text + "\n\n" + table_summary
    if len(context) > max_chars:
        # trim to last max_chars characters (prefer recent text)
        context = context[-max_chars:]
        # add an indicator
        context = "[CONTEXT TRIMMED]\n" + context
    return context


def ask_ollama(question: str, context: str, model: str = "mistral", timeout: int = 20) -> str:
    """
    Send prompt to local Ollama model. Returns the model text output.
    Uses subprocess to call: ollama run <model> and sends prompt via stdin.
    If Ollama is not available or there is an error, raises RuntimeError.
    """
    # Compose a safe prompt that instructs the model to answer only from context.
    prompt = (
        "You are a helpful assistant specialized in reading financial documents."
        " Use only the information provided in the CONTEXT to answer the question. If the information is not present, say 'Not present in document'."
        f"\n\nCONTEXT:\n{context}\n\nQUESTION: {question}\n\nAnswer concisely and reference the exact value or table row if possible."
    )

    try:
        # Call ollama: send prompt to stdin
        proc = subprocess.run(
            ["ollama", "run", model],
            input=prompt,
            text=True,
            capture_output=True,
            timeout=timeout,
        )
    except FileNotFoundError:
        # Ollama not found
        raise RuntimeError("Ollama command not found. Install Ollama and ensure it is in PATH.")
    except subprocess.TimeoutExpired:
        raise RuntimeError("Ollama call timed out.")

    if proc.returncode != 0:
        # include stderr in the error
        raise RuntimeError(f"Ollama returned non-zero exit code. stderr: {proc.stderr.strip()}")

    return proc.stdout.strip()


def simple_fallback_answer(question: str, context: str) -> str:
    """
    Very simple rule-based fallback: tries to locate numbers near keywords in context.
    Not a replacement for LLM, but usable if Ollama isn't available.
    """
    q = question.lower()
    # Keywords commonly asked
    keywords = {
        "revenue": ["revenue", "sales", "total revenue", "net sales"],
        "profit": ["profit", "net profit", "net income", "profit after tax", "pat"],
        "expense": ["expense", "expenses", "operating expense", "total expense"],
        "ebitda": ["ebitda"],
    }

    # find numeric tokens in context
    # numeric regex captures numbers like 1,234,567.89 or 1234567
    num_pattern = re.compile(r"([₹$€]?\s?[\d\.,]{2,})")
    lines = context.splitlines()
    for key, kws in keywords.items():
        for kw in kws:
            if kw in q:
                # search lines containing keyword
                candidates = [ln for ln in lines if kw in ln.lower()]
                for c in candidates:
                    match = num_pattern.search(c)
                    if match:
                        return f"Found in document: {c.strip()}"
                # if not found in same line, search neighbors
                idxs = [i for i, ln in enumerate(lines) if kw in ln.lower()]
                for i in idxs:
                    window = lines[max(0, i - 2): min(len(lines), i + 3)]
                    for w in window:
                        match = num_pattern.search(w)
                        if match:
                            return f"Found near '{kw}': {w.strip()}"
                return "Keyword found but no numeric value detected nearby in the extracted text."
    # default no info
    # as last attempt, return first numeric value in context
    m = num_pattern.search(context)
    if m:
        return f"Found numeric value in document: {m.group(1)} (context search)"
    return "No relevant numeric information could be found in the document."
