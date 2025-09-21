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
    data_summary: Optional[str] # Ringkasan statistik dari pandas

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
# 🤖 Agent 2: Classifier (Parameter Extractor)
# ==============================
def classifier_agent(state: AgentState):
    """
    Mengekstrak semua parameter yang mungkin dari pertanyaan untuk mendukung query multi-kriteria.
    Mode diatur langsung dari intent BERT.
    """
    try:
        q = state["question"].lower()
        bert_intent = state.get("bert_intent", "general_fallback")
        params = {}
        
        # Set mode langsung dari klasifikasi BERT
        state["mode"] = bert_intent

        def find_param(pattern, text, group=1):
            match = re.search(pattern, text)
            return match.group(group).strip() if match else None

        # Ekstraksi Area
        area_match = find_param(r'area\s+(rmk1|rmk2|fmd|rmp)', q)
        if area_match: params['area'] = area_match

        # Ekstraksi Status
        status_match = find_param(r'status\s+(done|belum|confirmed)', q)
        if status_match: params['status'] = status_match.capitalize()
        elif "sudah selesai" in q or "statusnya done" in q: params['status'] = "Done"
        elif "statusnya belum" in q: params['status'] = "Belum"

        # Ekstraksi Urgensi
        urgency_match = find_param(r'urgensi\s+(high|med|low)', q)
        if urgency_match: params['urgensi'] = urgency_match.capitalize()

        # Ekstraksi PIC
        pic_match = find_param(r'pic\s+([a-z\s]+?)(?=\sdi|\sdengan|\sarea|$)', q) or find_param(r'oleh\s+([a-z\s]+?)(?=\sdi|\sdengan|\sarea|$)', q)
        if pic_match: params['pic'] = pic_match
        
        # Ekstraksi HAC
        hac_match = find_param(r'hac\s+([a-z0-9\s\.\-_]+)', q)
        if hac_match: params['hac'] = hac_match

        # Ekstraksi Tanggal dan Rentang Tanggal
        start_date_match = find_param(r'antara\s+([0-9\-\/]+)', q)
        end_date_match = find_param(r'sampai\s+([0-9\-\/]+)', q)
        if start_date_match and end_date_match:
            params['tanggal_awal'] = start_date_match
            params['tanggal_akhir'] = end_date_match
        else:
            date_match = find_param(r'tanggal\s+([0-9\-\/]+)', q)
            if date_match: params['tanggal'] = date_match
        
        # Ekstraksi Bulan
        month_match = re.search(r'bulan\s+(januari|februari|maret|april|mei|juni|juli|agustus|september|oktober|november|desember)', q)
        if month_match:
            month_map = {'januari': 1, 'februari': 2, 'maret': 3, 'april': 4, 'mei': 5, 'juni': 6, 'juli': 7, 'agustus': 8, 'september': 9, 'oktober': 10, 'november': 11, 'desember': 12}
            params['bulan'] = month_map[month_match.group(1)]

        state["params"] = params
            
    except Exception as e:
        st.error(f"Error di Classifier Agent: {e}")
        state["mode"] = "general_fallback" 
        state["answer"] = "Maaf, terjadi kesalahan saat mengekstrak detail dari pertanyaan Anda."
    
    return state

# ==============================
# 🤖 Agent 3: Data Describer
# ==============================
def data_describer_agent(state: AgentState):
    """
    Membuat ringkasan statistik deskriptif dari data berdasarkan parameter
    untuk memberikan konteks yang padat kepada agent selanjutnya.
    """
    try:
        if df is None:
            state["data_summary"] = "Data tidak tersedia untuk dianalisis."
            return state

        params = state.get("params", {})
        summary_parts = []
        
        PARAM_TO_COLUMN_MAP = {
            'area': 'Area',
            'status': 'Status',
            'urgensi': 'Level Urgensi',
            'pic': 'PIC'
        }
        
        relevant_cols = [key for key in PARAM_TO_COLUMN_MAP.keys() if key in params]

        if len(relevant_cols) >= 2:
            # Jika ada 2+ parameter relevan, buat crosstab.
            col1_key, col2_key = relevant_cols[0], relevant_cols[1]
            col1_name, col2_name = PARAM_TO_COLUMN_MAP[col1_key], PARAM_TO_COLUMN_MAP[col2_key]
            
            cross_tab = pd.crosstab(df[col1_name], df[col2_name])
            summary_parts.append(f"Ringkasan Silang antara '{col1_name}' dan '{col2_name}':\n" + cross_tab.to_string())
        
        elif len(relevant_cols) == 1:
            col_key = relevant_cols[0]
            col_name = PARAM_TO_COLUMN_MAP[col_key]
            counts = df[col_name].value_counts().to_string()
            summary_parts.append(f"Distribusi untuk kolom '{col_name}':\n" + counts)

        if not summary_parts:
             state["data_summary"] = "Tidak ada ringkasan statistik yang relevan untuk pertanyaan ini."
        else:
            state["data_summary"] = "\n\n".join(summary_parts)

    except Exception as e:
        st.error(f"Error di Data Describer Agent: {e}")
        state["data_summary"] = "Gagal membuat ringkasan data."
        
    return state

# ==============================
# 🤖 Agent 4: Retriever
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

def retriever_agent(state: AgentState):
    """Mengambil data berdasarkan mode dan parameter, mendukung query multi-kriteria dan statistik."""
    try:
        mode = state.get("mode", "general_fallback")
        params = state.get("params", {})
        
        if df is None:
            state["context"] = "⚠️ Data tidak tersedia."
            return state
        
        if mode == "filter_data_query":
            filtered_df = apply_filters(df, params)
            if filtered_df.empty:
                state["context"] = "Tidak ada data yang cocok dengan kombinasi kriteria Anda."
            else:
                state["context"] = filtered_df.head(15).to_csv(index=False)
        
        elif mode == "statistical_query":
            filtered_df = apply_filters(df, params) # Terapkan filter dasar dulu (misal: bulan)
            q = state['question'].lower()
            
            if "trend" in q and "bulan" in q:
                filtered_df['Bulan'] = filtered_df['Tanggal'].dt.to_period('M')
                trend = filtered_df.groupby('Bulan').size().sort_index()
                context_text = "Tren override per bulan (dengan filter aktif):\n"
                for period, count in trend.items():
                    context_text += f"- {period}: {count} override\n"
                state["context"] = context_text
            elif "terbanyak" in q:
                target_col = "Area" # Default
                if "pic" in q: target_col = "PIC"
                elif "alasan" in q: target_col = "Alasan"

                if filtered_df.empty:
                    state["context"] = f"Tidak ada data untuk dianalisis dengan filter yang diberikan."
                    return state

                counts = filtered_df[target_col].value_counts()
                context_text = f"Peringkat '{target_col}' dengan override terbanyak (dengan filter aktif):\n"
                for item, count in counts.head(5).items():
                    context_text += f"- {item}: {count} override\n"
                state["context"] = context_text
            else:
                state["context"] = "Maaf, jenis analisis statistik ini belum didukung."

        elif mode == "comparison_query":
            q = state['question'].lower()
            items_to_compare = list(set(re.findall(r'(rmk1|rmk2|fmd|rmp)', q)))
            if len(items_to_compare) < 2:
                state["context"] = "Mohon sebutkan setidaknya dua area (RMK1, RMK2, FMD, RMP) untuk dibandingkan."
                return state
            
            filtered_df = apply_filters(df, params)
            context_text = "Hasil perbandingan jumlah override (dengan filter aktif):\n"
            for area in items_to_compare:
                count = len(filtered_df[filtered_df['Area'].str.contains(area, case=False, na=False)])
                context_text += f"- Area {area.upper()}: {count} override\n"
            state["context"] = context_text

        elif mode == "latest_override":
            latest_entry = df.sort_values(by="Tanggal", ascending=False).iloc[0]
            state["context"] = f"Override terbaru:\n- Tanggal: {latest_entry['Tanggal']}\n- HAC: {latest_entry['HAC']}\n- Alasan: {latest_entry['Alasan']}\n- Override oleh: {latest_entry['Override by']}"
        
        elif mode == "oldest_override":
            oldest_entry = df.sort_values(by="Tanggal", ascending=True).iloc[0]
            state["context"] = f"Override terdahulu:\n- Tanggal: {oldest_entry['Tanggal']}\n- HAC: {oldest_entry['HAC']}\n- Alasan: {oldest_entry['Alasan']}\n- Override oleh: {oldest_entry['Override by']}"
        
        elif mode == "list_entities_query":
            q = state['question'].lower()
            col_map = {"pic": "PIC", "manager": "Manager", "override by": "Override by", "area": "Area"}
            target_col = "PIC" 
            for keyword, col_name in col_map.items():
                if keyword in q:
                    target_col = col_name
                    break
            entities = df[target_col].dropna().unique()
            state["context"] = f"Daftar untuk '{target_col}':\n- {', '.join(entities)}"

        elif mode == 'count_query':
            filtered_df = apply_filters(df, params)
            state['context'] = f"Ditemukan total {len(filtered_df)} data override dengan kriteria yang diberikan."

        else: # general_fallback
             state['context'] = df.head().to_csv(index=False)

    except Exception as e:
        st.error(f"Error di Retriever Agent: {e}")
        state["context"] = "Maaf, terjadi kesalahan saat mengambil dan memproses data."

    return state

# ==============================
# 🤖 Agent 5: Answer (LLM + Token Log)
# ==============================
def answer_agent(state: AgentState):
    """
    Agent untuk menjawab pertanyaan user menggunakan LLM, dengan konteks tambahan dari Data Describer.
    """
    if state.get("answer"):
        return state
    
    direct_answer_indicators = ["daftar untuk", "hasil perbandingan", "tren override", "peringkat", "total", "adalah", "terbaru:", "terdahulu:"]

    context_lower = state.get("context", "").lower()
    if any(indicator in context_lower for indicator in direct_answer_indicators) or "tidak ada data" in context_lower:
        state["answer"] = state["context"]
        return state

    try:
        user_question = state["question"]
        context = state["context"]
        data_summary = state.get("data_summary", "Tidak ada.")
        model = "gpt-4o-mini"
        
        full_prompt = (
            f"Anda adalah asisten data. Jawab pertanyaan pengguna berdasarkan informasi yang diberikan.\n"
            f"Anda memiliki dua sumber informasi:\n"
            f"1. Ringkasan Statistik: Ini adalah data agregat yang mungkin bisa langsung menjawab pertanyaan tentang jumlah atau distribusi.\n"
            f"2. Potongan Data Mentah: Ini adalah beberapa baris contoh data yang relevan.\n\n"
            f"--- RINGKASAN STATISTIK ---\n{data_summary}\n\n"
            f"--- POTONGAN DATA MENTAH ---\n{context}\n\n"
            f"--- PERTANYAAN ---\n{user_question}\n\n"
            f"Prioritaskan jawaban dari Ringkasan Statistik jika memungkinkan (misalnya untuk menjawab 'berapa banyak'). Gunakan data mentah untuk detail tambahan jika perlu."
        )

        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "Anda adalah asisten data yang cerdas. Jawab pertanyaan hanya dari konteks yang ada. Sajikan jawaban dalam format yang mudah dibaca, gunakan poin-poin jika perlu."},
                {"role": "user", "content": full_prompt},
            ],
        )
        output_text = completion.choices[0].message.content
        state["answer"] = output_text.strip()
        
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
workflow.add_node("data_describer", data_describer_agent)
workflow.add_node("retriever", retriever_agent)
workflow.add_node("answer", answer_agent)

workflow.set_entry_point("bert_router")
workflow.add_edge("bert_router", "classifier")
workflow.add_edge("classifier", "data_describer")
workflow.add_edge("data_describer", "retriever")
workflow.add_edge("retriever", "answer")
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
        with st.spinner("Para agent sedang bekerja..."):
            result = app.invoke({"question": user_question})

            st.subheader("📊 Proses Kerja Agent")
            col1, col2 = st.columns(2)

            with col1:
                if "bert_classification" in st.session_state:
                    st.info("**Agent 1: BERT Router**")
                    st.json(st.session_state["bert_classification"])
                
                st.info("**Agent 4: Retriever**")
                st.text_area("Konteks Data Mentah", value=result.get("context", "Tidak ada konteks yang diambil."), height=250)
                
            with col2:
                st.info("**Agent 2: Classifier (Rules)**")
                st.json({
                    "mode_terpilih": result.get("mode", "N/A"),
                    "parameter_terekstrak": result.get("params", {})
                })
                
                st.info("**Agent 3: Data Describer**")
                st.text_area("Ringkasan Statistik", value=result.get("data_summary", "Tidak ada ringkasan."), height=250)

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
                    st.info("Belum ada penggunaan token LLM yang tercatat untuk pertanyaan ini.")

