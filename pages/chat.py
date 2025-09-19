import streamlit as st
import pandas as pd
import requests
import io
from PyPDF2 import PdfReader
import openai

# Konfigurasi LLM
client = openai.OpenAI(
    base_url="https://api.llm7.io/v1",
    api_key="O8i6UL79AeMnQonLiWFa0Irb2JjsIiJvuWFEjlpw9bHC3OyYECHmRjH0ttilenRaVnswQXtElhLxX91ecSMzLmSe0IoR7EngK60BCIXQE11EfDLUC7TPxGsaSQQ="  # ganti dengan key kalau punya
)

st.set_page_config(page_title="LLM Chat & File Reader", layout="wide")

st.title("🤖 LLM Chat & File Reader")

# Pilihan mode
mode = st.radio("Pilih mode:", ["💬 Chat Biasa", "📂 Chat dengan File (via URL)"])

# -----------------------
# MODE 1: Chat biasa
# -----------------------
if mode == "💬 Chat Biasa":
    st.subheader("Chat dengan LLM")

    # Session state untuk chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    user_input = st.text_input("Ketik pesan kamu:")

    if st.button("Kirim") and user_input.strip():
        st.session_state.chat_history.append({"role": "user", "content": user_input})

        with st.spinner("Menghubungi LLM..."):
            response = client.chat.completions.create(
                model="gpt-4.1-nano-2025-04-14",
                messages=st.session_state.chat_history
            )
            answer = response.choices[0].message.content
            st.session_state.chat_history.append({"role": "assistant", "content": answer})

    # Tampilkan chat history
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.markdown(f"**🧑 Kamu:** {msg['content']}")
        else:
            st.markdown(f"**🤖 LLM:** {msg['content']}")

# -----------------------
# MODE 2: Chat dengan File
# -----------------------
elif mode == "📂 Chat dengan File (via URL)":
    st.subheader("Baca file dari URL + Chat dengan LLM")

    file_url = st.text_input("Masukkan URL file (PDF, Excel, CSV):")

    if file_url:
        try:
            response = requests.get(file_url)
            response.raise_for_status()
            content = response.content

            extracted_text = ""

            if file_url.endswith(".pdf"):
                pdf = PdfReader(io.BytesIO(content))
                extracted_text = "\n".join([page.extract_text() or "" for page in pdf.pages])
            elif file_url.endswith((".xlsx", ".xls")):
                df = pd.read_excel(io.BytesIO(content))
                st.subheader("📊 Preview Excel:")
                st.dataframe(df.head())
                extracted_text = df.to_csv(index=False)
            elif file_url.endswith(".csv"):
                df = pd.read_csv(io.BytesIO(content))
                st.subheader("📊 Preview CSV:")
                st.dataframe(df.head())
                extracted_text = df.to_csv(index=False)
            else:
                extracted_text = content.decode("utf-8", errors="ignore")

            st.success("✅ File berhasil dibaca!")

            user_question = st.text_area("Tanyakan sesuatu tentang file ini:")

            if st.button("💬 Tanya LLM") and user_question.strip():
                with st.spinner("Menghubungi LLM..."):
                    llm_response = client.chat.completions.create(
                        model="gpt-4.1-nano-2025-04-14",
                        messages=[
                            {"role": "system", "content": "You are an assistant that analyzes documents."},
                            {"role": "user", "content": f"File content:\n{extracted_text[:4000]}"},
                            {"role": "user", "content": user_question}
                        ]
                    )
                    answer = llm_response.choices[0].message.content
                    st.markdown("### 🤖 Jawaban LLM:")
                    st.write(answer)

        except Exception as e:
            st.error(f"Gagal membaca file: {e}")
