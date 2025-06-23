import streamlit as st
import cv2
import numpy as np
import pandas as pd
import uuid
import re
from datetime import datetime
from ultralytics import YOLO
import easyocr
import torch
import os

# Konfigurasi awal
MODEL_PATH = "./Model_best/plat_nomor_best.pt"

# Setup device CUDA jika tersedia
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Inisialisasi EasyOCR
try:
    reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())
except Exception as e:
    st.error(f"Gagal memuat EasyOCR: {str(e)}")
    st.stop()


# Load model YOLO ke device yang sesuai
try:
    # Cek apakah file model ada
    if not os.path.exists(MODEL_PATH):
        st.error(f"File model tidak ditemukan di path: {MODEL_PATH}")
        st.info("Pastikan file 'plat_nomor_best.pt' berada di dalam folder 'Model_best' di direktori yang sama dengan aplikasi Anda.")
        st.stop()
    model = YOLO(MODEL_PATH).to(DEVICE)
except Exception as e:
    st.error(f"Gagal memuat model YOLO: {str(e)}")
    st.stop()

# Konfigurasi antarmuka Streamlit
st.set_page_config(page_title="ANPR System", layout="wide")
st.title("🚘 Automatic Number Plate Recognition System")

# Sidebar untuk pengaturan
with st.sidebar:
    st.header("⚙️ Pengaturan Sistem")

    res_options = {
        'HD (1280x720)': (1280, 720),
        'Full HD (1920x1080)': (1920, 1080),
        '480p (640x480)': (640, 480),
        'Custom': 'custom'
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
    st.markdown("**Developed by:** NADHIF & Tim")

# Inisialisasi session state
if 'detections' not in st.session_state:
    st.session_state.detections = []
if 'run_processing' not in st.session_state:
    st.session_state.run_processing = False

# Regex untuk validasi plat nomor Indonesia (disempurnakan)
PLATE_REGEX = re.compile(
    r'^[A-Z]{1,2}\s?(\d{1,4})\s?[A-Z]{1,3}$'
)

def validate_plate(text):
    """Membersihkan dan memvalidasi teks OCR agar sesuai format plat nomor."""
    clean_text = ''.join(filter(str.isalnum, text)).upper()
    
    # Menghapus teks yang kemungkinan adalah tanggal STNK
    if re.match(r'\d{4}', clean_text) or re.match(r'\d{2}\d{2}', clean_text):
        return None

    match = PLATE_REGEX.match(clean_text)
    if match:
        # Rekonstruksi format plat dengan spasi
        parts = re.split(r'(\d+)', clean_text)
        if len(parts) >= 3:
            return f"{parts[0]} {parts[1]} {''.join(parts[2:])}".strip()
        return clean_text
    return None

def resize_frame(frame, target_width, target_height):
    """Mengubah ukuran frame dengan menjaga aspek rasio."""
    return cv2.resize(frame, (target_width, target_height))

def process_frame(frame, target_size):
    """Memproses satu frame video untuk deteksi dan OCR."""
    frame_resized = resize_frame(frame, target_size[0], target_size[1])
    results = model(frame_resized, verbose=False, device=DEVICE)[0]
    newly_detected_plates = []
    
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        conf = float(box.conf[0])
        if conf >= conf_threshold:
            cv2.rectangle(frame_resized, (x1, y1), (x2, y2), (0, 255, 0), 2)
            try:
                plate_roi = frame_resized[y1:y2, x1:x2]
                ocr_results = reader.readtext(plate_roi, text_threshold=ocr_confidence)
                
                validated_plate = None
                if ocr_results:
                    # Coba validasi setiap hasil OCR
                    for _, text, _ in ocr_results:
                        plate = validate_plate(text)
                        if plate:
                            validated_plate = plate
                            break # Ambil hasil valid pertama
                    
                    # Jika tidak ada yang valid, coba gabungkan
                    if not validated_plate:
                        combined_text = ' '.join([res[1] for res in ocr_results])
                        validated_plate = validate_plate(combined_text)

                if validated_plate:
                    cv2.putText(frame_resized, validated_plate, (x1, y1-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2, cv2.LINE_AA)
                    
                    # Cek apakah plat sudah terdeteksi sebelumnya
                    if not any(d['Plat'] == validated_plate for d in st.session_state.detections):
                        detection_data = {
                            'ID': str(uuid.uuid4())[:6],
                            'Waktu': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            'Plat': validated_plate,
                            'Kepercayaan Deteksi': f"{conf:.2%}",
                            'Resolusi': f"{target_size[0]}x{target_size[1]}",
                            'Posisi': f"({x1},{y1}) - ({x2},{y2})"
                        }
                        newly_detected_plates.append(detection_data)
            except Exception as e:
                if show_debug:
                    st.warning(f"Error processing ROI: {str(e)}")
    
    if newly_detected_plates:
        st.session_state.detections.extend(newly_detected_plates)
    
    return frame_resized

# --- MAIN LAYOUT ---

col1, col2 = st.columns([3, 1])
with col1:
    video_source = st.radio("Pilih Sumber Video:", 
                            ("Webcam", "Upload Video"), 
                            horizontal=True, key="video_source_selector")

with col2:
    if st.button("🔄 Reset Deteksi"):
        st.session_state.detections = []
        st.rerun() # FIX: Menggunakan st.rerun() yang modern

video_placeholder = st.empty()
data_placeholder = st.empty()

# FIX: Inisialisasi cap ke None
cap = None 

if video_source == "Webcam":
    try:
        cap = cv2.VideoCapture(0) # Menggunakan indeks 0 untuk webcam default
        if not cap.isOpened():
            st.error("Gagal mengakses webcam. Pastikan webcam terhubung dan tidak digunakan oleh aplikasi lain.")
            st.session_state.run_processing = False
        else:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, res_width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, res_height)
            st.session_state.run_processing = True
    except Exception as e:
        st.error(f"Error saat membuka webcam: {e}")
        st.session_state.run_processing = False

elif video_source == "Upload Video":
    uploaded_file = st.file_uploader("Upload file video", type=["mp4", "avi", "mov", "mkv"])
    if uploaded_file:
        # Simpan file yang di-upload ke file sementara
        temp_file_path = os.path.join(".", uploaded_file.name)
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        cap = cv2.VideoCapture(temp_file_path)
        st.session_state.run_processing = True
    else:
        st.session_state.run_processing = False
        video_placeholder.info("Silakan upload file video untuk memulai pemrosesan.")

# Loop pemrosesan utama
while st.session_state.run_processing and cap and cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        st.success("Pemrosesan video selesai!")
        st.session_state.run_processing = False
        break
    
    processed_frame = process_frame(frame, (res_width, res_height))
    # Konversi BGR (OpenCV) ke RGB (Streamlit)
    video_placeholder.image(processed_frame[:, :, ::-1], 
                            channels="RGB", 
                            use_container_width=True)
    
    if st.session_state.detections:
        # Tampilkan data unik terakhir berdasarkan plat
        df = pd.DataFrame(st.session_state.detections).drop_duplicates('Plat', keep='last')
        data_placeholder.dataframe(
            df[['ID', 'Waktu', 'Plat', 'Kepercayaan Deteksi', 'Resolusi']],
            height=300,
            use_container_width=True
        )

# --- CLEANUP AND FINAL ACTIONS ---

# FIX: Lakukan release hanya jika cap telah diinisialisasi
if cap is not None:
    cap.release()

# Tombol download akan selalu muncul jika ada data
if st.session_state.detections:
    # Hapus duplikat sebelum download
    final_df = pd.DataFrame(st.session_state.detections).drop_duplicates('Plat', keep='last')
    csv = final_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Data CSV",
        data=csv,
        file_name=f"deteksi_plat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )

# Tampilkan informasi debug jika dicentang
if show_debug:
    st.subheader("ℹ️ Informasi Debug")
    debug_col1, debug_col2 = st.columns(2)
    with debug_col1:
        st.markdown("**Status Sistem:**")
        # FIX: Cek apakah cap isOpened sebelum mengakses propertinya
        is_cap_opened = cap and cap.isOpened() if cap else False
        st.json({
            "Resolusi Aktif": f"{res_width}x{res_height}",
            "Total Deteksi Unik": len(pd.DataFrame(st.session_state.detections).drop_duplicates('Plat')),
            "FPS (Sumber)": f"{cap.get(cv2.CAP_PROP_FPS):.2f}" if is_cap_opened else "N/A",
            "Device": DEVICE
        })
    
    with debug_col2:
        st.markdown("**Konfigurasi Kamera:**")
        is_cap_opened = cap and cap.isOpened() if cap else False
        st.json({
            "Frame Width (Actual)": cap.get(cv2.CAP_PROP_FRAME_WIDTH) if is_cap_opened else "N/A",
            "Frame Height (Actual)": cap.get(cv2.CAP_PROP_FRAME_HEIGHT) if is_cap_opened else "N/A"
        })
