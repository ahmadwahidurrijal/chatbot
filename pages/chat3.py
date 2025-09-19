import streamlit as st
import pandas as pd
import openai
import tiktoken
from langgraph.graph import StateGraph, END
from typing import TypedDict

# ==============================
# 🔑 Konfigurasi LLM Dummy (ubah sesuai provider)
# ==============================
# Anda dapat mengganti base_url ini dengan endpoint OpenAI asli jika diperlukan
client = openai.OpenAI(
    base_url="https://api.llm7.io/v1",
    api_key="unused"
)

# ==============================
# 🔢 Token Counter
# ==============================
def count_tokens(text: str, model: str = "gpt-o4-mini-2025-04-16") -> int:
    """Menghitung jumlah token dalam teks."""
    try:
        enc = tiktoken.encoding_for_model(model)
    except KeyError:
        enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))

# ==============================
# Load Data dari Google Sheets
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
# State untuk LangGraph
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

# ==============================
# Agent 1: Classifier
# ==============================
def classifier_agent(state: AgentState):
    """Mengklasifikasikan pertanyaan user untuk menentukan mode dan parameter."""
    q = state["question"].lower()
    state["params"] = {}
    
    # Fungsi pembantu untuk mencari parameter
    def find_param(pattern, text, group=1):
        match = re.search(pattern, text)
        return match.group(group) if match else None

    # Logika klasifikasi untuk pertanyaan kompleks
    # Filter No
    if "total override sampai nomor" in q:
        state["mode"] = "count_until_no"
        state["params"]["no"] = find_param(r'nomor\s+(\d+)', q)
    elif "data override dengan nomor" in q:
        state["mode"] = "by_no"
        state["params"]["no"] = find_param(r'nomor\s+(\d+)', q)
    
    # Filter Tanggal
    elif "berapa override pada tanggal" in q:
        state["mode"] = "count_by_date"
        state["params"]["tanggal"] = find_param(r'tanggal\s+([0-9\-\/]+)', q)
    elif "override apa saja antara" in q:
        state["mode"] = "date_range"
        state["params"]["tanggal_awal"] = find_param(r'antara\s+([0-9\-\/]+)', q)
        state["params"]["tanggal_akhir"] = find_param(r'sampai\s+([0-9\-\/]+)', q)
    elif "override pada tanggal" in q:
        state["mode"] = "by_date"
        state["params"]["tanggal"] = find_param(r'tanggal\s+([0-9\-\/]+)', q)
    
    # Filter HAC
    elif "override dengan hac" in q:
        state["mode"] = "by_hac"
        state["params"]["hac"] = find_param(r'hac\s+([a-z0-9\s\.\-_]+)', q)
    elif "memiliki hac tertentu" in q:
        state["mode"] = "hac_not_empty"

    # Filter Area
    elif "area mana saja yang paling banyak override" in q:
        state["mode"] = "most_overrides_by_area"
    elif "override di area" in q:
        state["mode"] = "by_area"
        state["params"]["area"] = find_param(r'area\s+(rmk1|rmk2|fmd|rmp)', q)

    # Filter Alasan
    elif "alasan-nya belum jelas" in q or "tidak diketahui" in q:
        state["mode"] = "alasan_unknown"
    elif "override dengan alasan" in q:
        state["mode"] = "by_alasan"
        state["params"]["alasan"] = find_param(r'alasan\s+([a-z\s]+)', q)

    # Filter Status
    elif "jumlah override dengan status" in q and "belum" in q:
        state["mode"] = "count_by_status"
        state["params"]["status"] = "Belum"
    elif "sudah selesai" in q or "statusnya done" in q:
        state["mode"] = "by_status"
        state["params"]["status"] = "Done"
    elif "override dengan status" in q:
        state["mode"] = "by_status"
        state["params"]["status"] = find_param(r'status\s+(done|belum|confirmed)', q)

    # Filter Notif
    elif "belum punya nomor notif" in q:
        state["mode"] = "notif_empty"
    elif "nomor notif" in q:
        state["mode"] = "by_notif"
        state["params"]["notif"] = find_param(r'notif\s+(\d+)', q)
    
    # Filter PIC
    elif "belum memiliki pic" in q:
        state["mode"] = "pic_empty"
    elif "ditangani oleh" in q:
        state["mode"] = "by_pic"
        state["params"]["pic"] = find_param(r'oleh\s+([a-z\s]+)', q)

    # Filter Level Urgensi
    elif "jumlah override dengan urgensi" in q:
        state["mode"] = "count_by_urgency"
        state["params"]["urgensi"] = find_param(r'urgensi\s+(high|med|low)', q)
    elif "override dengan urgensi" in q:
        state["mode"] = "by_urgency"
        state["params"]["urgensi"] = find_param(r'urgensi\s+(high|med|low)', q)

    # Filter Mitigasi Resiko
    elif "sudah memiliki mitigasi resiko" in q:
        state["mode"] = "mitigasi_not_empty"
    elif "belum ada mitigasi resiko" in q:
        state["mode"] = "mitigasi_empty"

    # Filter Action
    elif "action-nya masih kosong" in q:
        state["mode"] = "action_empty"
    elif "override dengan action" in q:
        state["mode"] = "by_action"
        state["params"]["action"] = find_param(r'action\s+([a-z\s]+)', q)

    # Filter Override by
    elif "belum di-override oleh siapapun" in q:
        state["mode"] = "override_by_empty"
    elif "di-override oleh" in q:
        state["mode"] = "by_override_by"
        state["params"]["nama"] = find_param(r'oleh\s+([a-z\s]+)', q)

    # Filter Manager
    elif "belum disetujui manager" in q:
        state["mode"] = "manager_empty"
    elif "disetujui oleh manager" in q:
        state["mode"] = "by_manager"
        state["params"]["nama"] = find_param(r'oleh manager\s+([a-z\s]+)', q)

    # Logika klasifikasi umum dan fallback
    elif any(w in q for w in ["override terbaru", "override terkini", "siapa yang override terbaru", "alasan override terbaru"]):
        state["mode"] = "latest_override"
    elif any(w in q for w in ["berapa", "jumlah", "banyak"]):
        state["mode"] = "count"
    elif any(w in q for w in ["status", "done", "belum", "confirmed"]):
        state["mode"] = "status"
    elif "area" in q or any(w in q for w in ["rmk1", "rmk2", "fmd", "rmp"]):
        state["mode"] = "area"
    elif any(w in q for w in ["tanggal terbaru", "terbaru", "terkini"]):
        state["mode"] = "latest_date"
    elif "override" in q and "hac" in q:
        state["mode"] = "override_by_hac"
    elif any(w in q for w in ["override by", "override oleh", "siapa yang override", "siapa override", "override"]):
        state["mode"] = "who_override"
    elif any(w in q for w in ["manager", "siapa manager", "managernya siapa"]):
        state["mode"] = "who_manager"
    else:
        state["mode"] = "teks"
    
    return state

# ==============================
# Agent 2: Retriever
# ==============================
def retriever_agent(state: AgentState):
    """Mengambil konteks data berdasarkan mode yang diklasifikasikan."""
    mode = state["mode"]
    params = state["params"]
    q = state["question"].lower()

    if df is None:
        state["context"] = "⚠️ Data tidak tersedia."
        return state
    
    filtered_df = pd.DataFrame()

    # Logika retrieval untuk pertanyaan spesifik
    if mode == "count_until_no":
        no = int(params["no"]) if params.get("no") else None
        if no:
            count = len(df[df["No"] <= no])
            state["context"] = f"Total override sampai nomor {no} adalah {count}."
            return state
    elif mode == "by_no":
        no = int(params["no"]) if params.get("no") else None
        if no:
            filtered_df = df[df["No"] == no]
    elif mode == "count_by_date":
        tanggal = params.get("tanggal")
        if tanggal:
            count = len(df[df["Tanggal"].dt.strftime("%Y-%m-%d") == tanggal])
            state["context"] = f"Ada {count} override pada tanggal {tanggal}."
            return state
    elif mode == "date_range":
        start_date = params.get("tanggal_awal")
        end_date = params.get("tanggal_akhir")
        if start_date and end_date:
            filtered_df = df[(df["Tanggal"] >= start_date) & (df["Tanggal"] <= end_date)]
    elif mode == "by_date":
        tanggal = params.get("tanggal")
        if tanggal:
            filtered_df = df[df["Tanggal"].dt.strftime("%Y-%m-%d") == tanggal]
    elif mode == "by_hac":
        hac = params.get("hac")
        if hac:
            filtered_df = df[df["HAC"].astype(str).str.contains(hac, case=False, na=False)]
    elif mode == "hac_not_empty":
        filtered_df = df[df["HAC"].notna()]
    elif mode == "most_overrides_by_area":
        counts = df['Area'].value_counts().nlargest(3)
        context_text = "Area dengan override terbanyak:\n"
        for area, count in counts.items():
            context_text += f"- {area}: {count} override\n"
        state["context"] = context_text
        return state
    elif mode == "by_area":
        area = params.get("area")
        if area:
            filtered_df = df[df["Area"].str.contains(area, case=False, na=False)]
    elif mode == "alasan_unknown":
        filtered_df = df[df["Alasan"].astype(str).str.contains("tidak diketahui|belum jelas", case=False, na=False)]
    elif mode == "by_alasan":
        alasan = params.get("alasan")
        if alasan:
            filtered_df = df[df["Alasan"].astype(str).str.contains(alasan, case=False, na=False)]
    elif mode == "count_by_status":
        status = params.get("status")
        if status:
            count = len(df[df["Status"].astype(str).str.contains(status, case=False, na=False)])
            state["context"] = f"Jumlah override dengan status '{status}' adalah {count}."
            return state
    elif mode == "by_status":
        status = params.get("status")
        if status:
            filtered_df = df[df["Status"].astype(str).str.contains(status, case=False, na=False)]
    elif mode == "notif_empty":
        filtered_df = df[df["Notif"].isna()]
    elif mode == "by_notif":
        notif = params.get("notif")
        if notif:
            filtered_df = df[df["Notif"].astype(str) == str(notif)]
    elif mode == "pic_empty":
        filtered_df = df[df["PIC"].isna()]
    elif mode == "by_pic":
        pic = params.get("pic")
        if pic:
            filtered_df = df[df["PIC"].astype(str).str.contains(pic, case=False, na=False)]
    elif mode == "count_by_urgency":
        urgency = params.get("urgensi")
        if urgency:
            count = len(df[df["Level Urgensi"].astype(str).str.contains(urgency, case=False, na=False)])
            state["context"] = f"Jumlah override dengan urgensi '{urgency}' adalah {count}."
            return state
    elif mode == "by_urgency":
        urgency = params.get("urgensi")
        if urgency:
            filtered_df = df[df["Level Urgensi"].astype(str).str.contains(urgency, case=False, na=False)]
    elif mode == "mitigasi_not_empty":
        filtered_df = df[df["Mitigasi Resiko"].notna()]
    elif mode == "mitigasi_empty":
        filtered_df = df[df["Mitigasi Resiko"].isna()]
    elif mode == "action_empty":
        filtered_df = df[df["Action"].isna()]
    elif mode == "by_action":
        action = params.get("action")
        if action:
            filtered_df = df[df["Action"].astype(str).str.contains(action, case=False, na=False)]
    elif mode == "override_by_empty":
        filtered_df = df[df["Override by"].isna()]
    elif mode == "by_override_by":
        nama = params.get("nama")
        if nama:
            filtered_df = df[df["Override by"].astype(str).str.contains(nama, case=False, na=False)]
    elif mode == "manager_empty":
        filtered_df = df[df["Manager"].isna()]
    elif mode == "by_manager":
        nama = params.get("nama")
        if nama:
            filtered_df = df[df["Manager"].astype(str).str.contains(nama, case=False, na=False)]
    
    # Logika retrieval umum dan fallback
    elif mode == "latest_override":
        filtered_df = df.dropna(subset=["Override by"]).sort_values(by="Tanggal", ascending=False)
        if not filtered_df.empty:
            latest_entry = filtered_df.iloc[0]
            context_text = f"Override terbaru:\n"
            context_text += f"Tanggal: {latest_entry['Tanggal']}\n"
            context_text += f"HAC: {latest_entry['HAC']}\n"
            context_text += f"Alasan: {latest_entry['Alasan']}\n"
            context_text += f"Override oleh: {latest_entry['Override by']}\n"
            state["context"] = context_text
            return state
        else:
            state["context"] = "Tidak ada data override yang ditemukan."
            return state
    elif mode == "who_override":
        overrides = df["Override by"].dropna().unique()
        if len(overrides) > 0:
            state["context"] = f"Daftar Override by: {', '.join(overrides)}"
        else:
            state["context"] = "Tidak ada data 'Override by' yang ditemukan."
        return state
    elif mode == "who_manager":
        managers = df["Manager"].dropna().unique()
        if len(managers) > 0:
            state["context"] = f"Daftar Manager: {', '.join(managers)}"
        else:
            state["context"] = "Tidak ada data 'Manager' yang ditemukan."
        return state
    elif mode == "override_by_hac":
        hac_kw = q.split("hac")[-1].strip()
        filtered_df = df[df["HAC"].astype(str).str.contains(hac_kw, case=False, na=False)]
        if filtered_df.empty:
            state["context"] = f"Tidak ditemukan HAC {hac_kw}"
            return state
        overrides = filtered_df["Override by"].dropna().unique()
        state["context"] = f"Override HAC {hac_kw}: {', '.join(overrides) if len(overrides) else 'Tidak ada'}"
        return state
    else:
        filtered_df = df.head(5)

    # Finalisasi konteks
    if filtered_df.empty:
        state["context"] = "Tidak ada data yang cocok dengan kriteria."
    else:
        state["context"] = filtered_df.head(10).to_csv(index=False)
    
    return state

# ==============================
# Agent 3: Answer (LLM + Token Log)
# ==============================
def answer_agent(state: AgentState):
    """
    Agent untuk menjawab pertanyaan user menggunakan LLM.
    Menerima state dari LangGraph.
    """
    user_question = state["question"]
    context = state["context"]
    model = "gpt-4o-mini"
    
    # Prompt yang lebih lengkap untuk LLM
    full_prompt = (
        f"Berdasarkan konteks data berikut, jawab pertanyaan pengguna.\n"
        f"Jika konteks tidak relevan atau tidak cukup untuk menjawab, katakan saja.\n\n"
        f"Konteks Data:\n{context}\n\n"
        f"Pertanyaan:\n{user_question}"
    )

    try:
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "Anda adalah agen data yang bermanfaat. Jawablah hanya berdasarkan konteks yang diberikan. Jangan pernah membuat informasi baru."},
                {"role": "user", "content": full_prompt},
            ],
        )

        output_text = completion.choices[0].message.content
        
        # Simpan jawaban di state
        state["answer"] = output_text.strip()
        
        # Hitung dan simpan token usage di session state Streamlit
        input_tokens = count_tokens(full_prompt, model)
        output_tokens = count_tokens(output_text, model)
        st.session_state["last_token_log"] = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens
        }

    except Exception as e:
        state["answer"] = f"⚠️ Error di answer_agent: {str(e)}"
    
    return state

# ==============================
# LangGraph Workflow
# ==============================
workflow = StateGraph(AgentState)
workflow.add_node("classifier", classifier_agent)
workflow.add_node("retriever", retriever_agent)
workflow.add_node("answer", answer_agent)

workflow.set_entry_point("classifier")
workflow.add_edge("classifier", "retriever")
workflow.add_edge("retriever", "answer")
workflow.add_edge("answer", END)

app = workflow.compile()

# ==============================
# Streamlit UI
# ==============================
st.set_page_config(page_title="Nested Hybrid Agent", layout="wide")
st.title("🤖 Nested Hybrid Agent + Token Counter")

if data_loaded:
    st.success("✅ Data berhasil dibaca dari Google Sheets")
    st.dataframe(df)

user_question = st.text_area("Tanyakan sesuatu tentang data ini:")

if st.button("💬 Tanya Agent") and user_question.strip():
    # Panggil LangGraph dan tangani hasilnya
    result = app.invoke({"question": user_question})
    st.subheader("📄 Konteks (Agent 2)")
    st.text(result["context"])
    st.subheader("🤖 Jawaban (Agent 3)")
    st.write(result["answer"])

    if "last_token_log" in st.session_state:
        st.subheader("📊 Penggunaan Token")
        st.json(st.session_state["last_token_log"])
