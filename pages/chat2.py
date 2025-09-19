import streamlit as st
import pandas as pd
import requests
import io
from PyPDF2 import PdfReader
import openai

# Konfigurasi LLM
client = openai.OpenAI(
    base_url="https://api.llm7.io/v1",
    #api_key="O8i6UL79AeMnQonLiWFa0Irb2JjsIiJvuWFEjlpw9bHC3OyYECHmRjH0ttilenRaVnswQXtElhLxX91ecSMzLmSe0IoR7EngK60BCIXQE11EfDLUC7TPxGsaSQQ=" # ganti dengan token dari https://token.llm7.io/ agar limit lebih longgar
    api_key="unused"
)

st.set_page_config(page_title="LLM Chat & File Reader", layout="wide")

st.title("🤖 LLM Chat & File Reader (with URL Table Support)")

mode = st.radio("Pilih mode:", ["💬 Chat Biasa", "📂 Chat dengan File/URL"])

# ----------------------
# MODE 1: Chat biasa
# ----------------------
if mode == "💬 Chat Biasa":
    st.subheader("Chat dengan LLM")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    user_input = st.text_input("Ketik pesan kamu:")

    if st.button("Kirim") and user_input.strip():
        st.session_state.chat_history.append({"role": "user", "content": user_input})

        try:
            with st.spinner("Menghubungi LLM..."):
                response = client.chat.completions.create(
                    model="gpt-4.1-nano-2025-04-14",
                    messages=st.session_state.chat_history
                )
                answer = response.choices[0].message.content
                st.session_state.chat_history.append({"role": "assistant", "content": answer})
        except Exception as e:
            st.error(f"Error: {e}")

    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.markdown(f"**🧑 Kamu:** {msg['content']}")
        else:
            st.markdown(f"**🤖 LLM:** {msg['content']}")

# ----------------------
# MODE 2: Chat dengan File/URL
# ----------------------
else:
    st.subheader("Baca file atau tabel dari URL + Chat dengan LLM")

    file_url = st.text_input("Masukkan URL (PDF, Excel, CSV, atau halaman web dengan tabel):")

    if file_url:
        extracted_text = ""

        try:
            if file_url.endswith(".pdf"):
                content = requests.get(file_url).content
                pdf = PdfReader(io.BytesIO(content))
                extracted_text = "\n".join([page.extract_text() or "" for page in pdf.pages])

            elif file_url.endswith((".xlsx", ".xls")):
                content = requests.get(file_url).content
                df = pd.read_excel(io.BytesIO(content))
                st.subheader("📊 Preview Excel:")
                st.dataframe(df.head())
                extracted_text = df.to_csv(index=False)

            elif file_url.endswith(".csv"):
                content = requests.get(file_url).content
                df = pd.read_csv(io.BytesIO(content))
                st.subheader("📊 Preview CSV:")
                st.dataframe(df.head())
                extracted_text = df.to_csv(index=False)

            else:
                # Coba parse tabel dari HTML
                try:
                    tables = pd.read_html(file_url)
                    if tables:
                        df = tables[0]
                        st.subheader("📊 Preview Tabel dari HTML:")
                        st.dataframe(df.head())
                        extracted_text = df.to_csv(index=False)
                    else:
                        extracted_text = requests.get(file_url).text[:5000]
                        st.warning("Tidak ada tabel, hanya ambil teks HTML.")
                except Exception:
                    extracted_text = requests.get(file_url).text[:5000]
                    st.warning("Tidak bisa parse tabel, hanya ambil teks HTML.")

            st.success("✅ Data berhasil dibaca!")

            user_question = st.text_area("Tanyakan sesuatu tentang data ini:")

            if st.button("💬 Tanya LLM") and user_question.strip():
                with st.spinner("Menghubungi LLM..."):
                    llm_response = client.chat.completions.create(
                        model="gpt-4.1-nano-2025-04-14",
                        messages=[
                            {"role": "system", "content": "You are an assistant that analyzes tabular data or documents."},
                            {"role": "user", "content": f"Data content:\n{extracted_text[:4000]}"},
                            {"role": "user", "content": user_question}
                        ]
                    )
                    answer = llm_response.choices[0].message.content
                    st.markdown("### 🤖 Jawaban LLM:")
                    st.write(answer)

        except Exception as e:
            st.error(f"Gagal membaca: {e}")
