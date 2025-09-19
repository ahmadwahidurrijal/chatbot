import streamlit as st
import pandas as pd
import requests
import io
from PyPDF2 import PdfReader
import openai
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import tiktoken

# -------------------
# Fungsi hitung token
# -------------------
def count_tokens(messages, model="gpt-4.1-nano-2025-04-14"):
    try:
        enc = tiktoken.encoding_for_model(model)
    except:
        enc = tiktoken.get_encoding("cl100k_base")
    return sum(len(enc.encode(msg["content"])) for msg in messages)

# -------------------
# Konfigurasi LLM
# -------------------
client = openai.OpenAI(
    base_url="https://api.llm7.io/v1",
    api_key="unused"
)

st.set_page_config(page_title="Nested Hybrid Agent", layout="wide")
st.title("🤖 Nested Hybrid Retriever untuk Monitoring Interlock")

# -------------------
# Load data Google Sheet
# -------------------
sheet_id = "1zhpo2eIfCTZ_ZkoJoyhFFAHmgDYOIQ-8"
sheet_gid = "806961825"
csv_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={sheet_gid}"

try:
    df = pd.read_csv(csv_url)
    df["Tanggal"] = pd.to_datetime(df["Tanggal"], errors="coerce")
    st.success("✅ Data berhasil dibaca!")
    st.dataframe(df)
except Exception as e:
    st.error(f"Gagal membaca Google Sheets: {e}")
    df = None

# Agent 1: Classifier
# -------------------
def classifier_agent(question):
    q = question.lower()

    # --- Count / Jumlah ---
    if any(w in q for w in ["berapa", "jumlah", "count", "banyak"]):
        return "count"

    # --- Tanggal ---
    elif any(w in q for w in ["terbaru", "paling baru", "tanggal terakhir", "latest"]):
        return "tanggal"

    # --- Status ---
    elif any(w in q for w in ["status", "belum", "done", "confirmed", "selesai"]):
        return "status"

    # --- Level Urgensi ---
    elif any(w in q for w in ["urgensi", "high", "med", "low", "priority", "prioritas"]):
        return "urgensi"

    # --- Area ---
    elif any(w in q for w in ["area", "rmk1", "rmk2", "fmd", "rmp", "lokasi", "plant", "unit"]):
        return "area"


    # --- HAC ---
    elif "hac" in q:
        return "hac"

    # --- Alasan ---
    elif any(w in q for w in ["alasan", "penyebab", "error", "sensor"]):
        return "alasan"

    # --- Notif ---
    elif "notif" in q or "notification" in q or "id" in q:
        return "notif"

    # --- PIC ---
    elif any(w in q for w in ["pic", "person", "penanggung jawab"]):
        return "pic"

    # --- Mitigasi ---
    elif any(w in q for w in ["mitigasi", "resiko", "risk"]):
        return "mitigasi"

    # --- Action ---
    elif any(w in q for w in ["action", "tindakan"]):
        return "action"

    # --- Override ---
    elif "override" in q:
        return "override"
    # --- Override by + HAC ---
    elif "override" in q and "hac" in q:
        return "override_by_hac"


    # --- Manager ---
    elif any(w in q for w in ["manager", "atasan"]):
        return "manager"

    # --- Default fallback ---
    else:
        return "teks"


# -------------------
# Agent 2: Retriever
# -------------------
def retriever_agent(mode, question, df):
    q = question.lower()

    if mode == "count":
        # Mapping kolom ke keyword khusus
        col_keywords = {
            "Status": ["belum", "done", "confirmed", "selesai"],
            "Level Urgensi": ["high", "med", "low"],
            "Area": ["rmk1", "rmk2", "fmd", "rmp"],
        }

        df_filtered = df.copy()
        applied_filters = []

        for col in df.columns:
            if col in col_keywords:
                for kw in col_keywords[col]:
                    if kw in q:
                        df_filtered = df_filtered[df_filtered[col].astype(str).str.contains(kw, case=False, na=False)]
                        applied_filters.append(f"{col} contains '{kw}'")
                        break  # stop di keyword pertama yg cocok

        # Hitung hasil akhir
        count = len(df_filtered)
        if applied_filters:
            return f"Jumlah baris dengan kondisi ({' AND '.join(applied_filters)}) = {count}"
        else:
            return f"Total baris dataset = {len(df)}"



    # --- Tanggal ---
    elif mode == "tanggal":
        latest = df.sort_values("Tanggal", ascending=False).head(5)
        return latest.to_csv(index=False)

    # --- Status ---
    elif mode == "status":
        if "belum" in q:
            filtered = df[df["Status"].str.contains("Belum", case=False, na=False)]
        elif "done" in q:
            filtered = df[df["Status"].str.contains("Done", case=False, na=False)]
        elif "confirmed" in q:
            filtered = df[df["Status"].str.contains("Confirmed", case=False, na=False)]
        else:
            filtered = df
        return filtered.head(5).to_csv(index=False)

    # --- Urgensi ---
    elif mode == "urgensi":
        if "high" in q:
            filtered = df[df["Level Urgensi"].str.contains("High", case=False, na=False)]
        elif "med" in q:
            filtered = df[df["Level Urgensi"].str.contains("Med", case=False, na=False)]
        elif "low" in q:
            filtered = df[df["Level Urgensi"].str.contains("Low", case=False, na=False)]
        else:
            filtered = df
        return filtered.head(5).to_csv(index=False)

    # --- Area ---
    elif mode == "area":
        if "rmk1" in q:
            filtered = df[df["Area"].str.contains("RMK1", case=False, na=False)]
        elif "rmk2" in q:
            filtered = df[df["Area"].str.contains("RMK2", case=False, na=False)]
        elif "fmd" in q:
            filtered = df[df["Area"].str.contains("FMD", case=False, na=False)]
        elif "rmp" in q:
            filtered = df[df["Area"].str.contains("RMP", case=False, na=False)]
        else:
            filtered = df
        return filtered.head(5).to_csv(index=False)


    # --- HAC ---
    elif mode == "hac":
        keyword = q.replace("hac", "").strip()
        filtered = df[df["HAC"].str.contains(keyword, case=False, na=False)] if keyword else df
        return filtered.head(5).to_csv(index=False)

    # --- Alasan ---
    elif mode == "alasan":
        keyword = q.replace("alasan", "").replace("penyebab", "").strip()
        filtered = df[df["Alasan"].str.contains(keyword, case=False, na=False)] if keyword else df
        return filtered.head(5).to_csv(index=False)

    # --- Notif ---
    elif mode == "notif":
        keyword = q.replace("notif", "").replace("notification", "").strip()
        filtered = df[df["Notif"].astype(str).str.contains(keyword, case=False, na=False)] if keyword else df
        return filtered.head(5).to_csv(index=False)

    # --- PIC ---
    elif mode == "pic":
        keyword = q.replace("pic", "").replace("penanggung jawab", "").strip()
        filtered = df[df["PIC"].astype(str).str.contains(keyword, case=False, na=False)] if keyword else df
        return filtered.head(5).to_csv(index=False)

    # --- Mitigasi ---
    elif mode == "mitigasi":
        keyword = q.replace("mitigasi", "").replace("resiko", "").strip()
        filtered = df[df["Mitigasi Resiko"].astype(str).str.contains(keyword, case=False, na=False)] if keyword else df
        return filtered.head(5).to_csv(index=False)

    # --- Action ---
    elif mode == "action":
        keyword = q.replace("action", "").replace("tindakan", "").strip()
        filtered = df[df["Action"].astype(str).str.contains(keyword, case=False, na=False)] if keyword else df
        return filtered.head(5).to_csv(index=False)

    # --- Override ---
    elif mode == "override":
        keyword = q.replace("override", "").strip()
        filtered = df[df["Override by"].astype(str).str.contains(keyword, case=False, na=False)] if keyword else df
        return filtered.head(5).to_csv(index=False)

    # --- Manager ---
    elif mode == "manager":
        keyword = q.replace("manager", "").replace("atasan", "").strip()
        filtered = df[df["Manager"].astype(str).str.contains(keyword, case=False, na=False)] if keyword else df
        return filtered.head(5).to_csv(index=False)
    # --- Override by + HAC ---
    elif mode == "override_by_hac":
        # cari HAC
        hac_kw = q.split("hac")[-1].strip()
        filtered = df[df["HAC"].astype(str).str.contains(hac_kw, case=False, na=False)]
        
        if filtered.empty:
            return f"Tidak ditemukan data HAC {hac_kw}"
        
        overrides = filtered["Override by"].dropna().unique()
        if len(overrides) == 0:
            return f"Tidak ada yang melakukan override pada HAC {hac_kw}"
        else:
            return f"Override pada HAC {hac_kw} dilakukan oleh: {', '.join(overrides)}"

    # --- General Q&A: "siapa/apa <kolom> pada <kolom2>=<keyword>" ---
    elif any(w in q for w in ["siapa", "apa"]):
        col_map = {
            "status": "Status",
            "urgensi": "Level Urgensi",
            "area": "Area",
            "hac": "HAC",
            "alasan": "Alasan",
            "notif": "Notif",
            "pic": "PIC",
            "mitigasi": "Mitigasi Resiko",
            "action": "Action",
            "override": "Override by",
            "manager": "Manager",
        }

        target_col = None
        filter_col = None
        keyword = None

        # Deteksi kolom target
        for k, v in col_map.items():
            if k in q:
                target_col = v
                break

        # Deteksi kolom filter + keyword
        for k, v in col_map.items():
            if k in q and v != target_col:
                filter_col = v
                # ambil kata setelah k
                parts = q.split(k)
                if len(parts) > 1:
                    keyword = parts[-1].strip()
                break

        if target_col and filter_col and keyword:
            filtered = df[df[filter_col].astype(str).str.contains(keyword, case=False, na=False)]
            if filtered.empty:
                return f"Tidak ditemukan data {target_col} pada {filter_col} mengandung '{keyword}'"
            else:
                values = filtered[target_col].dropna().unique()
                return f"{target_col} pada {filter_col} '{keyword}' = {', '.join(values)}"
        else:
            return "Pertanyaan tidak jelas kolom target/filter-nya."

    # --- Default fallback TF-IDF ---
    else:
        row_texts = df.astype(str).agg(" ".join, axis=1)
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(row_texts)
        query_vec = vectorizer.transform([question])
        similarity = cosine_similarity(query_vec, tfidf_matrix).flatten()
        top_idx = similarity.argsort()[-5:][::-1]
        relevant_rows = df.iloc[top_idx]
        return relevant_rows.to_csv(index=False)
    
    

# -------------------
# Agent 3: Answer Agent (LLM)
# -------------------
def answer_agent(context, question):
    messages = [
        {"role": "system", "content": "You are an assistant that analyzes monitoring interlock data."},
        {"role": "user", "content": f"Data relevan:\n{context}"},
        {"role": "user", "content": question}
    ]
    token_count = count_tokens(messages)
    st.info(f"🔢 Token input: {token_count}")

    with st.spinner("Menghubungi LLM..."):
        llm_response = client.chat.completions.create(
            model="gpt-4.1-nano-2025-04-14",
            messages=messages
        )
    return llm_response.choices[0].message.content

# -------------------
# Chat UI
# -------------------
if df is not None:
    user_question = st.text_area("Tanyakan sesuatu tentang data ini:")

    if st.button("💬 Tanya LLM") and user_question.strip():
        # Agent 1: klasifikasi pertanyaan
        mode = classifier_agent(user_question)
        st.write(f"🤖 Agent 1 memilih mode: **{mode}**")

        # Agent 2: ambil data relevan
        context = retriever_agent(mode, user_question, df)
        st.text_area("📄 Data relevan (Agent 2):", context, height=200)

        # Agent 3: LLM answer
        answer = answer_agent(context, user_question)
        st.markdown("### 🤖 Jawaban LLM (Agent 3):")
        st.write(answer)
