import streamlit as st
import cv2
import numpy as np
import pandas as pd
import uuid
import re
from datetime import datetime, date
from ultralytics import YOLO
import easyocr
import torch
import os
import sqlite3

# --- KONFIGURASI ANTARMUKA STREAMLIT (HARUS DIJALANKAN PERTAMA) ---
st.set_page_config(page_title="ANPR System", layout="wide")
st.title("🚘 Automatic Number Plate Recognition System")

# --- KONFIGURASI DAN INISIALISASI ---

# Konfigurasi Awal
MODEL_PATH = "./Model_best/plat_nomor_best.pt"
DB_PATH = "anpr_data.db"
EVIDENCE_DIR = "data_bukti"

# Pastikan folder untuk menyimpan gambar bukti ada
os.makedirs(EVIDENCE_DIR, exist_ok=True)

# Setup device CUDA jika tersedia
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

@st.cache_resource
def load_easyocr():
    """Memuat model EasyOCR sekali dan menyimpannya di cache."""
    try:
        return easyocr.Reader(['en'], gpu=torch.cuda.is_available())
    except Exception as e:
        st.error(f"Gagal memuat EasyOCR: {str(e)}")
        st.stop()

@st.cache_resource
def load_yolo_model():
    """Memuat model YOLO sekali dan menyimpannya di cache."""
    try:
        if not os.path.exists(MODEL_PATH):
            st.error(f"File model tidak ditemukan di path: {MODEL_PATH}")
            st.info("Pastikan file 'plat_nomor_best.pt' berada di dalam folder 'Model_best' di direktori yang sama dengan aplikasi Anda.")
            st.stop()
        return YOLO(MODEL_PATH).to(DEVICE)
    except Exception as e:
        st.error(f"Gagal memuat model YOLO: {str(e)}")
        st.stop()

reader = load_easyocr()
model = load_yolo_model()

def init_db():
    """Inisialisasi database SQLite dan membuat tabel jika belum ada."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS detections (
            id TEXT PRIMARY KEY,
            waktu TIMESTAMP,
            plat TEXT,
            kepercayaan_deteksi TEXT,
            resolusi TEXT,
            posisi TEXT,
            status TEXT,
            path_gambar TEXT
        )
    """)
    conn.commit()
    conn.close()

# Panggil fungsi inisialisasi database saat aplikasi dimulai
init_db()

# Inisialisasi session state
if 'run_processing' not in st.session_state:
    st.session_state.run_processing = False
if 'blacklist' not in st.session_state:
    st.session_state.blacklist = ""
if 'whitelist' not in st.session_state:
    st.session_state.whitelist = ""

# --- SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Pengaturan Sistem")
    
    camera_index = st.number_input("Indeks Kamera", min_value=0, max_value=10, value=0, step=1, help="Ubah jika Anda memiliki lebih dari satu kamera. Kamera utama biasanya 0.")

    res_options = {
        'HD (1280x720)': (1280, 720), 'Full HD (1920x1080)': (1920, 1080),
        '480p (640x480)': (640, 480), 'Custom': 'custom'
    }
    selected_res = st.selectbox("Resolusi Video", list(res_options.keys()))
    if selected_res == 'Custom':
        custom_res = st.text_input("Masukkan resolusi (format: widthxheight)", "640x480")
        try:
            res_width, res_height = map(int, custom_res.split('x'))
        except:
            st.warning("Format resolusi tidak valid! Menggunakan default 640x480")
            res_width, res_height = 640, 480
    else:
        res_width, res_height = res_options[selected_res]

    conf_threshold = st.slider("Threshold Deteksi (%)", 1, 100, 45) / 100
    ocr_confidence = st.slider("Threshold OCR (%)", 1, 100, 30) / 100
    
    show_debug = st.checkbox("Tampilkan Informasi Debug")
    
    st.divider()
    
    st.header("🚦 Daftar Hitam & Putih")
    st.session_state.blacklist = st.text_area("Daftar Hitam (satu plat per baris)", value=st.session_state.blacklist)
    st.session_state.whitelist = st.text_area("Daftar Putih (satu plat per baris)", value=st.session_state.whitelist)

    st.divider()

# --- FUNGSI-FUNGSI UTAMA ---
PLATE_REGEX = re.compile(r'^[A-Z]{1,2}\s?(\d{1,4})\s?[A-Z]{1,3}$')

def get_lists():
    blacklist = {line.strip().upper().replace(" ", "") for line in st.session_state.blacklist.split('\n') if line.strip()}
    whitelist = {line.strip().upper().replace(" ", "") for line in st.session_state.whitelist.split('\n') if line.strip()}
    return blacklist, whitelist

def validate_plate(text):
    clean_text = ''.join(filter(str.isalnum, text)).upper()
    if re.match(r'\d{4}', clean_text) or re.match(r'\d{2}\d{2}', clean_text): return None
    match = PLATE_REGEX.match(clean_text)
    if match:
        parts = re.split(r'(\d+)', clean_text)
        return f"{parts[0]} {parts[1]} {''.join(parts[2:])}".strip() if len(parts) >= 3 else clean_text
    return None

def add_detection_to_db(data):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM detections WHERE plat = ?", (data['Plat'],))
    existing_record = cursor.fetchone()

    if existing_record is None:
        cursor.execute("""
            INSERT INTO detections (id, waktu, plat, kepercayaan_deteksi, resolusi, posisi, status, path_gambar)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (data['ID'], data['Waktu'], data['Plat'], data['Kepercayaan Deteksi'], data['Resolusi'], data['Posisi'], data['Status'], data['Path Gambar']))
        conn.commit()
        conn.close()
        return True
    conn.close()
    return False

def process_frame(frame, target_size):
    frame_resized = cv2.resize(frame, (target_size[0], target_size[1]))
    results = model(frame_resized, verbose=False, device=DEVICE)[0]
    blacklist, whitelist = get_lists()
    
    new_detection_made = False
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        conf = float(box.conf[0])
        if conf >= conf_threshold:
            plate_roi = frame_resized[y1:y2, x1:x2]
            ocr_results = reader.readtext(plate_roi, text_threshold=ocr_confidence, detail=0)

            validated_plate = None
            if ocr_results:
                combined_text = ' '.join(ocr_results)
                for text in ocr_results + [combined_text]:
                    plate = validate_plate(text)
                    if plate:
                        validated_plate = plate
                        break
            
            box_color = (0, 255, 0)
            status = "Normal"

            if validated_plate:
                plate_no_space = validated_plate.replace(" ", "")
                if plate_no_space in blacklist:
                    status = "Blacklist"
                    box_color = (0, 0, 255)
                elif plate_no_space in whitelist:
                    status = "Whitelist"
                    box_color = (255, 0, 0)

                cv2.rectangle(frame_resized, (x1, y1), (x2, y2), box_color, 2)
                cv2.putText(frame_resized, f"{validated_plate} [{status}]", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, box_color, 2, cv2.LINE_AA)
                
                timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                img_filename = f"{validated_plate.replace(' ', '_')}_{timestamp_str}.jpg"
                img_path = os.path.join(EVIDENCE_DIR, img_filename)
                
                detection_data = {
                    'ID': str(uuid.uuid4())[:8], 'Waktu': datetime.now(), 'Plat': validated_plate,
                    'Kepercayaan Deteksi': f"{conf:.2%}", 'Resolusi': f"{target_size[0]}x{target_size[1]}",
                    'Posisi': f"({x1},{y1})-({x2},{y2})", 'Status': status, 'Path Gambar': img_path
                }

                if add_detection_to_db(detection_data):
                    cv2.imwrite(img_path, plate_roi)
                    new_detection_made = True
                    if status == "Blacklist":
                        st.toast(f"🚨 PERINGATAN: Kendaraan Daftar Hitam terdeteksi! Plat: {validated_plate}", icon="🚨")

    return frame_resized, new_detection_made

def fetch_data(search_query=None, start_date=None, end_date=None):
    conn = sqlite3.connect(DB_PATH)
    query = "SELECT id, waktu, plat, status, kepercayaan_deteksi, path_gambar FROM detections"
    conditions = []
    params = []

    if search_query:
        conditions.append("plat LIKE ?")
        params.append(f"%{search_query}%")
    if start_date:
        conditions.append("waktu >= ?")
        params.append(start_date.strftime("%Y-%m-%d 00:00:00"))
    if end_date:
        conditions.append("waktu <= ?")
        params.append(end_date.strftime("%Y-%m-%d 23:59:59"))

    if conditions:
        query += " WHERE " + " AND ".join(conditions)
    
    query += " ORDER BY waktu DESC"
    
    df = pd.read_sql_query(query, conn, params=params)
    conn.close()
    return df

def style_df(df):
    def apply_style(row):
        color = ''
        if row.status == 'Blacklist':
            color = 'background-color: #ffcccc'
        elif row.status == 'Whitelist':
            color = 'background-color: #cceeff'
        return [color] * len(row)
    
    return df.style.apply(apply_style, axis=1)

# --- LAYOUT UTAMA ---
video_source = st.radio("Pilih Sumber Video:", ("Webcam", "Upload Video"), horizontal=True, key="video_source_selector")
video_placeholder = st.empty()
st.divider()

# --- AREA DATA DAN FILTER (Didefinisikan sebelum loop video) ---
st.header("📜 Data Kendaraan Tercatat")
filter1, filter2, filter3 = st.columns([2, 1, 1])
with filter1:
    search_term = st.text_input("Cari Plat Nomor", placeholder="Masukkan sebagian atau seluruh plat...")
with filter2:
    start_date_filter = st.date_input("Dari Tanggal", value=None)
with filter3:
    end_date_filter = st.date_input("Sampai Tanggal", value=None)

# Placeholder untuk tabel data dan tombol download
data_placeholder = st.empty()

# --- LOGIKA PEMROSESAN VIDEO DAN PEMBARUAN REAL-TIME ---
cap = None
if video_source == "Webcam":
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        st.error(f"Gagal mengakses kamera dengan indeks {camera_index}.")
        st.session_state.run_processing = False
    else:
        st.session_state.run_processing = True
elif video_source == "Upload Video":
    uploaded_file = st.file_uploader("Upload file video", type=["mp4", "avi", "mov", "mkv"])
    if uploaded_file:
        temp_file_path = os.path.join(".", uploaded_file.name)
        with open(temp_file_path, "wb") as f: f.write(uploaded_file.getbuffer())
        cap = cv2.VideoCapture(temp_file_path)
        st.session_state.run_processing = True

# Selalu tampilkan data terakhir, bahkan sebelum video dimulai
df_detections = fetch_data(search_term, start_date_filter, end_date_filter)
with data_placeholder.container():
    if not df_detections.empty:
        st.dataframe(style_df(df_detections), use_container_width=True)
        csv = df_detections.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Data (CSV)", data=csv,
            file_name=f"deteksi_plat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    else:
        st.info("Belum ada data deteksi. Mulai video untuk mencatat data.")

if st.session_state.run_processing and cap:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.success("Pemrosesan video selesai!")
            st.session_state.run_processing = False
            break
        
        processed_frame, new_detection_made = process_frame(frame, (res_width, res_height))
        video_placeholder.image(processed_frame[:, :, ::-1], channels="RGB", use_container_width=True)

        # Perbarui tabel HANYA jika ada deteksi baru untuk efisiensi
        if new_detection_made:
            df_detections = fetch_data(search_term, start_date_filter, end_date_filter)
            with data_placeholder.container():
                st.dataframe(style_df(df_detections), use_container_width=True)
                csv = df_detections.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Data (CSV)", data=csv,
                    file_name=f"deteksi_plat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )

    cap.release()
    if video_source == "Upload Video":
        st.rerun()

# --- AREA DEBUG ---
if show_debug:
    st.subheader("ℹ️ Informasi Debug")
    db_conn = sqlite3.connect(DB_PATH)
    total_records = pd.read_sql_query("SELECT COUNT(*) FROM detections", db_conn).iloc[0,0]
    db_conn.close()
    
    st.json({
        "Resolusi Aktif": f"{res_width}x{res_height}",
        "Total Catatan di DB": total_records,
        "Device": DEVICE,
        "Daftar Hitam Aktif": list(get_lists()[0]),
        "Daftar Putih Aktif": list(get_lists()[1])
    })
