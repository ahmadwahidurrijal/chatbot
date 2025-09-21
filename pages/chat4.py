import streamlit as st
import pandas as pd
import openai
import tiktoken
import re
from langgraph.graph import StateGraph, END
from typing import TypedDict, Optional
from sentence_transformers import SentenceTransformer, util
import torch
from datetime import datetime

# ==============================
# 🔑 Konfigurasi LLM Dummy (ubah sesuai provider)
# ==============================
# Anda dapat mengganti base_url ini dengan endpoint OpenAI asli jika diperlukan
client = openai.OpenAI(
    base_url="https://api.llm7.io/v1",
    api_key="unused"
)

# ==============================
# 🧠 Load Model BERT
# ==============================
@st.cache_resource
def load_bert_model():
    """Memuat model SentenceTransformer dan menyimpannya di cache Streamlit."""
    return SentenceTransformer('paraphrase-MiniLM-L6-v2')

bert_model = load_bert_model()

# ==============================
# 🔢 Token Counter & Logger
# ==============================
def count_tokens(text: str, model: str = "gpt-4o-mini") -> int:
    """Menghitung jumlah token dalam teks."""
    try:
        enc = tiktoken.encoding_for_model(model)
    except KeyError:
        enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))

def log_token_usage(question: str, token_log: dict):
    """Mencatat penggunaan token ke dalam file harian."""
    try:
        today = datetime.now().strftime("%Y-%m-%d")
        log_filename = f"token_log_{today}.txt"
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        log_entry = (
            f"--- Log at {timestamp} ---\n"
            f"Question: {question}\n"
            f"Input Tokens: {token_log.get('input_tokens', 0)}\n"
            f"Output Tokens: {token_log.get('output_tokens', 0)}\n"
            f"Total Tokens: {token_log.get('total_tokens', 0)}\n"
            f"---------------------------------\n\n"
        )
        
        with open(log_filename, 'a', encoding='utf-8') as f:
            f.write(log_entry)
        return log_filename
    except Exception as e:
        st.warning(f"Gagal menulis ke file log: {e}")
        return None

# ==============================
# 📄 Load Data dari Google Sheets
# ==============================
sheet_id = "1zhpo2eIfCTZ_ZkoJoyhFFAHmgDYOIQ-8"
sheet_gid = "806961825"
csv_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={sheet_gid}"

try:
    df = pd.read_csv(csv_url)
    df["Tanggal"] = pd.to_datetime(df["Tanggal"], errors="coerce")
    data_loaded = True
except Exception as e:
    st.error(f"Gagal membaca Google Sheets: {e}")
    df = None
    data_loaded = False

# ==============================
# 🛠️ Input Validation Utilities
# ==============================
def validate_date_format(date_string: str) -> Optional[datetime]:
    """Mencoba mem-parsing string tanggal. Mengembalikan objek datetime atau None."""
    if not date_string or not isinstance(date_string, str):
        return None
    try:
        parsed_date = pd.to_datetime(date_string, errors='coerce')
        return None if pd.isna(parsed_date) else parsed_date.to_pydatetime()
    except (ValueError, TypeError):
        return None

def safe_to_int(value: any) -> Optional[int]:
    """Mengonversi nilai ke integer dengan aman. Mengembalikan None jika gagal."""
    if value is None:
        return None
    try:
        return int(value)
    except (ValueError, TypeError):
        return None

# ==============================
# 📌 State untuk LangGraph
# ==============================
class AgentState(TypedDict):
    """
    Representasi state dalam LangGraph.
    Setiap node akan memodifikasi atau membaca dari state ini.
    """
    question: str
    mode: str
    params: dict
    context: str
    answer: str
    bert_intent: Optional[str] # Menyimpan hasil klasifikasi dari BERT

# ==============================
# 🤖 Agent 1: BERT Router
# ==============================
INTENTS = {
    "filter_data_query": "Mencari data override berdasarkan satu atau lebih kriteria spesifik (misalnya area, status, PIC, HAC, tanggal, urgensi).",
    "statistical_query": "Menanyakan pertanyaan statistik atau agregat (misalnya tren bulanan, area terbanyak dalam periode waktu).",
    "comparison_query": "Membandingkan jumlah atau data antara dua atau lebih kategori (misalnya RMK1 vs RMK2).",
    "list_entities_query": "Meminta daftar entitas unik dari sebuah kolom (misalnya daftar semua PIC, daftar manajer).",
    "latest_query": "Menampilkan data override yang paling baru atau terkini.",
    "oldest_query": "Menampilkan data override yang paling lama, terdahulu, atau terlampau.",
    "count_query": "Menghitung jumlah total data override.",
    "general_fallback": "Pertanyaan umum atau tidak spesifik yang memerlukan analisis lebih lanjut."
}

def bert_router_agent(state: AgentState):
    """
    Menggunakan BERT untuk klasifikasi semantik sebagai router awal.
    """
    try:
        question = state["question"]
        
        question_embedding = bert_model.encode(question, convert_to_tensor=True)
        intent_labels = list(INTENTS.keys())
        intent_descriptions = list(INTENTS.values())
        intent_embeddings = bert_model.encode(intent_descriptions, convert_to_tensor=True)
        
        cosine_scores = util.cos_sim(question_embedding, intent_embeddings)
        
        best_match_idx = torch.argmax(cosine_scores)
        best_match_score = cosine_scores[0][best_match_idx].item()
        best_match_intent = intent_labels[best_match_idx]
        
        state["bert_intent"] = best_match_intent
        st.session_state["bert_classification"] = {
            "intent_terdeteksi": best_match_intent,
            "skor_keyakinan": f"{best_match_score:.4f}",
            "deskripsi_intent": INTENTS[best_match_intent]
        }
    except Exception as e:
        st.error(f"Error di BERT Router Agent: {e}")
        state["bert_intent"] = "general_fallback"
        state["answer"] = "Maaf, terjadi kesalahan saat menganalisis pertanyaan Anda di tahap awal."

    return state

# ==============================
# 🤖 Agent 2: Classifier (Parameter Extractor & Mode Refiner)
# ==============================
def classifier_agent(state: AgentState):
    """
    Mengekstrak semua parameter dan menyempurnakan 'mode' berdasarkan kata kunci
    untuk memastikan tindakan yang paling spesifik dieksekusi.
    """
    try:
        q = state["question"].lower()
        bert_intent = state.get("bert_intent", "general_fallback")
        params = {}
        
        # Set mode awal dari klasifikasi BERT
        current_mode = bert_intent
        state["mode"] = current_mode

        def find_param(pattern, text, group=1):
            match = re.search(pattern, text)
            return match.group(group).strip() if match else None

        # Peta terstruktur untuk ekstraksi parameter
        PARAMETER_PATTERNS = {
            'area': [r'area\s+(rmk1|rmk2|fmd|rmp)', r'\b(fmd|rmk1|rmk2|rmp)\b'],
            'status': [r'status\s+(done|belum|confirmed)', r'\b(done|belum|confirmed)\b'],
            'urgensi': [r'urgensi\s+(high|med|low)', r'\b(high|med|low)\b'],
            'pic': [r'pic\s+([a-z\s]+?)(?=\sdi|\sdengan|\sarea|$)', r'oleh\s+([a-z\s]+?)(?=\sdi|\sdengan|\sarea|$)'],
            'hac': [r'hac\s+([a-z0-9\s\.\-_]+)'],
            'tanggal': [r'tanggal\s+([0-9\-\/]+)'],
            'tanggal_awal': [r'antara\s+([0-9\-\/]+)'],
            'tanggal_akhir': [r'sampai\s+([0-9\-\/]+)'],
        }

        for param_name, patterns in PARAMETER_PATTERNS.items():
            for pattern in patterns:
                match = find_param(pattern, q)
                if match:
                    if param_name in ['status', 'urgensi']:
                        params[param_name] = match.capitalize()
                    else:
                        params[param_name] = match
                    break
        
        if 'status' not in params:
            if "sudah selesai" in q or "statusnya done" in q: params['status'] = "Done"
            elif "statusnya belum" in q: params['status'] = "Belum"

        month_match = re.search(r'bulan\s+(januari|februari|maret|april|mei|juni|juli|agustus|september|oktober|november|desember)', q)
        if month_match:
            month_map = {'januari': 1, 'februari': 2, 'maret': 3, 'april': 4, 'mei': 5, 'juni': 6, 'juli': 7, 'agustus': 8, 'september': 9, 'oktober': 10, 'november': 11, 'desember': 12}
            params['bulan'] = month_map[month_match.group(1)]

        # --- Penyempurnaan Mode ---
        # Jika ada kata kunci tindakan yang lebih spesifik, ganti mode.
        # Ini menangani kasus "FMD terbaru" di mana BERT mungkin hanya memilih 'filter_data_query'.
        if "terbaru" in q or "terkini" in q:
            state["mode"] = "latest_query"
        elif "terlama" in q or "terdahulu" in q:
            state["mode"] = "oldest_query"
        elif "jumlah" in q or "berapa banyak" in q and bert_intent != 'comparison_query':
             state["mode"] = "count_query"

        state["params"] = params
            
    except Exception as e:
        st.error(f"Error di Classifier Agent: {e}")
        state["mode"] = "general_fallback" 
        state["answer"] = "Maaf, terjadi kesalahan saat mengekstrak detail dari pertanyaan Anda."
    
    return state

# ==============================
# 🤖 Agent 3: Data Processing Agent
# ==============================
def apply_filters(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """Menerapkan serangkaian filter ke DataFrame berdasarkan parameter yang diekstrak."""
    temp_df = df.copy()
    if params.get('area'):
        temp_df = temp_df[temp_df["Area"].str.contains(params['area'], case=False, na=False)]
    if params.get('status'):
        temp_df = temp_df[temp_df["Status"].str.contains(params['status'], case=False, na=False)]
    if params.get('urgensi'):
        temp_df = temp_df[temp_df["Level Urgensi"].str.contains(params['urgensi'], case=False, na=False)]
    if params.get('pic'):
        temp_df = temp_df[temp_df["PIC"].str.contains(params['pic'], case=False, na=False)]
    if params.get('hac'):
        temp_df = temp_df[temp_df["HAC"].astype(str).str.contains(params['hac'], case=False, na=False)]
    if params.get('tanggal'):
        tanggal = validate_date_format(params['tanggal'])
        if tanggal:
            temp_df = temp_df[temp_df['Tanggal'].dt.date == tanggal.date()]
    if params.get('tanggal_awal') and params.get('tanggal_akhir'):
        start_date = validate_date_format(params['tanggal_awal'])
        end_date = validate_date_format(params['tanggal_akhir'])
        if start_date and end_date:
            temp_df = temp_df[(temp_df['Tanggal'] >= start_date) & (temp_df['Tanggal'] <= end_date)]
    if params.get('bulan'):
        temp_df = temp_df[temp_df['Tanggal'].dt.month == params['bulan']]
    return temp_df

def data_processing_agent(state: AgentState):
    """
    Agen terpadu untuk memfilter, menganalisis, dan menyiapkan konteks atau jawaban langsung.
    Menerapkan logika fallback jika query spesifik tidak menghasilkan apa-apa.
    """
    try:
        mode = state.get("mode", "general_fallback")
        params = state.get("params", {})
        
        if df is None:
            state["answer"] = "⚠️ Data tidak tersedia."
            return state

        # 1. Selalu terapkan filter di awal
        filtered_df = apply_filters(df, params)
        q = state['question'].lower()
        
        # 2. Coba proses mode yang bisa memberikan jawaban langsung (direct answer)
        if mode == "statistical_query":
            if "trend" in q and "bulan" in q:
                if not filtered_df.empty:
                    filtered_df['Bulan'] = filtered_df['Tanggal'].dt.to_period('M')
                    trend = filtered_df.groupby('Bulan').size().sort_index()
                    answer_text = "Tren override per bulan (dengan filter aktif):\n"
                    for period, count in trend.items():
                        answer_text += f"- {period}: {count} override\n"
                    state["answer"] = answer_text
                    return state
            elif "terbanyak" in q or "paling umum" in q or "distribusi" in q:
                COLUMN_KEYWORDS = {"Area": ["area"],"Status": ["status"],"Level Urgensi": ["urgensi", "level urgensi"],"PIC": ["pic", "person in charge", "siapa"],"Alasan": ["alasan", "penyebab"],"Override by": ["override oleh"],"Manager": ["manager", "manajer"]}
                target_col = next((col for col, kw in COLUMN_KEYWORDS.items() if any(k in q for k in kw)), None)
                if not target_col:
                    state["answer"] = "Mohon sebutkan kolom apa yang ingin Anda lihat peringkatnya (misalnya 'area terbanyak', 'pic terbanyak')."
                    return state
                if not filtered_df.empty:
                    counts = filtered_df[target_col].value_counts()
                    answer_text = f"Peringkat '{target_col}' dengan override terbanyak (dengan filter aktif):\n"
                    for item, count in counts.head(5).items():
                        answer_text += f"- {item}: {count} override\n"
                    state["answer"] = answer_text
                    return state

        elif mode == "comparison_query":
            items_to_compare = list(set(re.findall(r'(rmk1|rmk2|fmd|rmp)', q)))
            if len(items_to_compare) >= 2:
                answer_text = "Hasil perbandingan jumlah override (dengan filter aktif):\n"
                for area in items_to_compare:
                    count = len(filtered_df[filtered_df['Area'].str.contains(area, case=False, na=False)])
                    answer_text += f"- Area {area.upper()}: {count} override\n"
                state["answer"] = answer_text
                return state
            
        elif mode == 'count_query':
            state['answer'] = f"Ditemukan total {len(filtered_df)} data override dengan kriteria yang diberikan."
            return state

        elif mode == "latest_query":
            if not filtered_df.empty:
                latest_entry = filtered_df.sort_values(by="Tanggal", ascending=False).iloc[0]
                state["answer"] = f"Override terbaru (dengan filter aktif):\n- Tanggal: {latest_entry['Tanggal']}\n- HAC: {latest_entry['HAC']}\n- Alasan: {latest_entry['Alasan']}\n- Override oleh: {latest_entry['Override by']}"
            else:
                state["answer"] = "Tidak ada data yang cocok dengan kriteria Anda untuk menemukan data terbaru."
            return state

        elif mode == "oldest_query":
            if not filtered_df.empty:
                oldest_entry = filtered_df.sort_values(by="Tanggal", ascending=True).iloc[0]
                state["answer"] = f"Override terdahulu (dengan filter aktif):\n- Tanggal: {oldest_entry['Tanggal']}\n- HAC: {oldest_entry['HAC']}\n- Alasan: {oldest_entry['Alasan']}\n- Override oleh: {oldest_entry['Override by']}"
            else:
                state["answer"] = "Tidak ada data yang cocok dengan kriteria Anda untuk menemukan data terdahulu."
            return state
        
        elif mode == "list_entities_query":
            col_map = {"pic": "PIC", "manager": "Manager", "override by": "Override by", "area": "Area"}
            target_col = next((col for kw, col in col_map.items() if kw in q), "PIC")
            entities = filtered_df[target_col].dropna().unique()
            if len(entities) > 0:
                state["answer"] = f"Daftar untuk '{target_col}' (dengan filter aktif):\n- {', '.join(entities)}"
            else:
                state["answer"] = f"Tidak ditemukan entitas '{target_col}' yang cocok dengan kriteria Anda."
            return state

        # 3. Logika Fallback / Default
        if not filtered_df.empty:
            summary_message = f"Ditemukan {len(filtered_df)} data yang cocok dengan kriteria Anda."
            if mode not in ["filter_data_query", "general_fallback"]:
                summary_message = f"Tidak dapat menemukan hasil spesifik untuk '{mode}', namun " + summary_message.lower()
            
            summary_message += " Berikut adalah 5 contoh teratas:"
            sample_data = filtered_df.head(5).to_csv(index=False)
            state["context"] = f"{summary_message}\n\n{sample_data}"
        else:
            state["answer"] = "Tidak ada data yang cocok dengan kombinasi kriteria Anda."

    except Exception as e:
        st.error(f"Error di Data Processing Agent: {e}")
        state["answer"] = "Maaf, terjadi kesalahan saat mengambil dan memproses data."

    return state


# ==============================
# 🤖 Agent 4: Answer (LLM + Token Log)
# ==============================
def answer_agent(state: AgentState):
    """
    Agent LLM yang hanya akan berjalan jika tidak ada jawaban langsung dari agent sebelumnya.
    """
    # *** INI ADALAH KUNCI PENGHEMATAN TOKEN ***
    # Jika state "answer" sudah diisi, berarti jawaban sudah final. Langsung kembalikan.
    if state.get("answer"):
        return state

    try:
        user_question = state["question"]
        context = state.get("context", "Tidak ada konteks yang relevan.")
        model = "gpt-4o-mini"
        
        full_prompt = (
            f"Anda adalah asisten data. Jawab pertanyaan pengguna berdasarkan ringkasan dan contoh data berikut.\n"
            f"Sajikan jawaban dalam format yang mudah dibaca dan ramah.\n\n"
            f"--- KONTEKS DATA ---\n{context}\n\n"
            f"--- PERTANYAAN ---\n{user_question}"
        )

        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "Anda adalah asisten data yang cerdas. Jawab pertanyaan hanya dari konteks yang ada. Jangan membuat informasi baru."},
                {"role": "user", "content": full_prompt},
            ],
        )
        output_text = completion.choices[0].message.content
        state["answer"] = output_text.strip()
        
        # Logging hanya terjadi jika LLM benar-benar dipanggil
        input_tokens = count_tokens(full_prompt, model)
        output_tokens = count_tokens(output_text, model)
        token_log = {
            "input_tokens": input_tokens, "output_tokens": output_tokens, "total_tokens": input_tokens + output_tokens
        }
        st.session_state["last_token_log"] = token_log
        log_token_usage(user_question, token_log)

    except openai.APIConnectionError as e:
        st.error(f"API Connection Error: {e.__cause__}")
        state["answer"] = "Maaf, terjadi masalah koneksi ke layanan AI."
    except openai.RateLimitError:
        st.error("API Rate Limit Error")
        state["answer"] = "Maaf, layanan sedang sibuk. Coba lagi nanti."
    except openai.APIStatusError as e:
        st.error(f"API Status Error: {e.status_code} - {e.response}")
        state["answer"] = f"Maaf, terjadi kesalahan pada layanan AI (Status: {e.status_code})."
    except Exception as e:
        st.error(f"Error di Answer Agent: {str(e)}")
        state["answer"] = "Maaf, terjadi kesalahan tak terduga saat memproses jawaban."
    
    return state

# ==============================
# 🔗 LangGraph Workflow
# ==============================
workflow = StateGraph(AgentState)

workflow.add_node("bert_router", bert_router_agent)
workflow.add_node("classifier", classifier_agent)
workflow.add_node("data_processor", data_processing_agent)
workflow.add_node("answer", answer_agent)

workflow.set_entry_point("bert_router")
workflow.add_edge("bert_router", "classifier")
workflow.add_edge("classifier", "data_processor")
workflow.add_edge("data_processor", "answer")
workflow.add_edge("answer", END)

app = workflow.compile()

# ==============================
# 🖥️ Streamlit UI
# ==============================
st.set_page_config(page_title="Nested Hybrid Agent", layout="wide")
st.title("🤖 Nested Hybrid Agent (BERT -> Rules -> LLM)")

if data_loaded:
    st.success("✅ Data berhasil dibaca dari Google Sheets")
    with st.expander("Lihat Data Mentah"):
        st.dataframe(df)
else:
    st.error("Gagal memuat data. Fitur chatbot mungkin tidak berfungsi dengan benar.")


user_question = st.text_area("Tanyakan sesuatu tentang data ini:", key="user_input", disabled=not data_loaded, placeholder="Contoh: tampilkan override di area RMK1 dengan status Belum dan urgensi High")

if st.button("💬 Tanya Agent", disabled=not data_loaded) and user_question.strip():
    if len(user_question) > 500:
        st.warning("Pertanyaan terlalu panjang. Harap persingkat pertanyaan Anda (maksimal 500 karakter).")
    else:
        # Reset log token sebelum pemanggilan baru
        if "last_token_log" in st.session_state:
            del st.session_state["last_token_log"]

        with st.spinner("Para agent sedang bekerja..."):
            result = app.invoke({"question": user_question})

            st.subheader("📊 Proses Kerja Agent")
            col1, col2 = st.columns(2)

            with col1:
                if "bert_classification" in st.session_state:
                    st.info("**Agent 1: BERT Router**")
                    st.json(st.session_state["bert_classification"])
                
                st.info("**Agent 3: Data Processor**")
                st.text_area("Konteks / Jawaban Langsung", value=result.get("context") or result.get("answer", "Tidak ada output."), height=250)
                
            with col2:
                st.info("**Agent 2: Classifier (Rules)**")
                st.json({
                    "mode_terpilih": result.get("mode", "N/A"),
                    "parameter_terekstrak": result.get("params", {})
                })
                
                if "last_token_log" in st.session_state:
                    st.info("**Agent 4: Answer (LLM)**")
                    st.write("LLM dipanggil untuk memproses jawaban.")
                else:
                    st.info("**Agent 4: Answer (LLM)**")
                    st.write("LLM dilewati (bypass) karena jawaban sudah final.")

            st.divider()
            st.subheader("💡 Jawaban Akhir")
            
            final_answer = result.get("answer", "Tidak ada jawaban yang bisa diberikan.")
            if any(w in final_answer.lower() for w in ["maaf", "tidak valid", "error", "gagal"]):
                 st.warning(final_answer)
            else:
                 st.success(final_answer)
            
            # Tampilkan bagian log token di bawah jawaban
            st.divider()
            with st.expander("📊 Detail Penggunaan Token & Log"):
                if "last_token_log" in st.session_state:
                    st.json(st.session_state["last_token_log"])
                    today = datetime.now().strftime("%Y-%m-%d")
                    log_filename = f"token_log_{today}.txt"
                    st.info(f"Penggunaan token ini telah dicatat dalam file: **{log_filename}**")
                else:
                    st.info("Tidak ada panggilan ke LLM, penggunaan token adalah 0.")

