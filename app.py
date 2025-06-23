import streamlit as st
import cv2
import numpy as np
import pandas as pd
import uuid
from datetime import datetime
from ultralytics import YOLO
import easyocr
import torch
import re

# Konfigurasi awal
MODEL_PATH = "./Model_best/plat_nomor_best.pt"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())

try:
    model = YOLO(MODEL_PATH)
    model.to(DEVICE)
except Exception as e:
    st.error(f"Gagal memuat model YOLO: {str(e)}")
    st.stop()

st.set_page_config(page_title="ANPR System", layout="wide")
st.title("🚘 Automatic Number Plate Recognition System")

with st.sidebar:
    st.header("⚙️ Pengaturan Sistem")
    st.subheader("ℹ️ Informasi Sistem")
    st.write(f"GPU Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        st.write(f"GPU Device: {torch.cuda.get_device_name(0)}")
    
    # Pilihan index kamera
    camera_index = st.number_input("Index Kamera", min_value=0, value=0, step=1)

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
            st.warning("Format tidak valid! Gunakan default 640x480.")
            res_width, res_height = 640, 480
    else:
        res_width, res_height = res_options[selected_res]

    conf_threshold = st.slider("Threshold Deteksi (%)", 1, 100, 45) / 100
    ocr_confidence = st.slider("Threshold OCR (%)", 1, 100, 20) / 100
    show_debug = st.checkbox("Tampilkan Informasi Debug")
    st.divider()
    st.markdown("**Developed by:** [Nama Anda]")

if 'detections' not in st.session_state:
    st.session_state.detections = []
if 'run_processing' not in st.session_state:
    st.session_state.run_processing = False

def resize_frame(frame, target_width, target_height):
    h, w = frame.shape[:2]
    aspect_ratio = w / h
    if target_width / target_height > aspect_ratio:
        new_height = target_height
        new_width = int(target_height * aspect_ratio)
    else:
        new_width = target_width
        new_height = int(target_width / aspect_ratio)
    return cv2.resize(frame, (new_width, new_height))

def format_plate_text(text):
    clean_text = ''.join(filter(str.isalnum, text)).strip().replace(" ", "").upper()
    match = re.match(r"^([A-Z]{1,2})(\d{1,4})([A-Z]{1,3})?$", clean_text)
    if match:
        return f"{match.group(1)} {match.group(2)} {match.group(3) or ''}".strip()
    return None

def process_frame(frame, target_size):
    frame = resize_frame(frame, target_size[0], target_size[1])
    results = model(frame, verbose=False, device=DEVICE)[0]
    current_detections = []

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        conf = float(box.conf[0])
        if conf >= conf_threshold:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            try:
                plate_roi = frame[y1:y2, x1:x2]
                gray = cv2.cvtColor(plate_roi, cv2.COLOR_BGR2GRAY)
                _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                ocr_result = reader.readtext(binary, detail=0, text_threshold=ocr_confidence)

                if ocr_result:
                    raw_text = ' '.join(ocr_result)
                    formatted_text = format_plate_text(raw_text)

                    if formatted_text:
                        # Cek duplikasi plat
                        existing = next((d for d in st.session_state.detections 
                                      if d['Plat'] == formatted_text and d['Waktu Keluar'] is None), None)
                        
                        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        
                        if not existing:
                            detection_data = {
                                'ID': str(uuid.uuid4())[:6],
                                'Waktu Masuk': timestamp,
                                'Waktu Keluar': None,
                                'Plat': formatted_text,
                                'Kepercayaan Deteksi': f"{conf:.2%}",
                                'Resolusi': f"{res_width}x{res_height}",
                                'Posisi': f"({x1},{y1}) - ({x2},{y2})"
                            }
                            st.session_state.detections.append(detection_data)
                        
                        cv2.putText(frame, formatted_text, (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            except Exception as e:
                if show_debug:
                    st.error(f"Error processing ROI: {str(e)}")

    return frame

col1, col2 = st.columns([3, 1])
with col1:
    video_source = st.radio("Pilih Sumber Video:", ("Webcam", "Upload Video"), horizontal=True)
with col2:
    if st.button("🔄 Reset Deteksi"):
        st.session_state.detections = []
        st.experimental_rerun()

video_placeholder = st.empty()
data_placeholder = st.empty()

if video_source == "Webcam":
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        st.error(f"Tidak dapat mengakses kamera dengan index {camera_index}")
        st.stop()
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, res_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, res_height)
    st.session_state.run_processing = True

elif video_source == "Upload Video":
    uploaded_file = st.file_uploader("Upload file video", type=["mp4", "avi", "mov"])
    if uploaded_file:
        temp_file = f"temp_{uploaded_file.name}"
        with open(temp_file, "wb") as f:
            f.write(uploaded_file.read())
        cap = cv2.VideoCapture(temp_file)
        st.session_state.run_processing = True
    else:
        st.session_state.run_processing = False

while st.session_state.run_processing:
    ret, frame = cap.read()
    if not ret:
        if video_source == "Webcam":
            st.error("Gagal mengakses webcam")
        else:
            st.success("Pemrosesan video selesai!")
        st.session_state.run_processing = False
        break

    processed_frame = process_frame(frame, (res_width, res_height))
    video_placeholder.image(processed_frame[:, :, ::-1], channels="RGB", use_column_width=True)

    if st.session_state.detections:
        df = pd.DataFrame(st.session_state.detections)
        data_placeholder.dataframe(
            df[['ID', 'Waktu Masuk', 'Plat', 'Kepercayaan Deteksi', 'Resolusi']],
            height=300,
            use_container_width=True
        )

cap.release()

# Bagian untuk menandai waktu keluar
if st.session_state.detections:
    st.subheader("Catatan Keluar")
    
    # Filter plat yang belum keluar
    active_plates = [d for d in st.session_state.detections if d['Waktu Keluar'] is None]
    
    if active_plates:
        selected_plate = st.selectbox(
            "Pilih Plat yang akan ditandai keluar:",
            options=[d['Plat'] for d in active_plates]
        )
        
        if st.button("Tandai Keluar"):
            for d in st.session_state.detections:
                if d['Plat'] == selected_plate and d['Waktu Keluar'] is None:
                    d['Waktu Keluar'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            st.experimental_rerun()
    
    # Tampilkan tabel lengkap
    st.dataframe(
        pd.DataFrame(st.session_state.detections)[
            ['ID', 'Plat', 'Waktu Masuk', 'Waktu Keluar', 'Kepercayaan Deteksi']
        ],
        use_container_width=True
    )
    
    # Download data
    csv = pd.DataFrame(st.session_state.detections).to_csv(index=False).encode()
    st.download_button("📥 Download Data CSV", data=csv, file_name="deteksi_plat.csv", mime="text/csv")

if show_debug:
    st.subheader("ℹ️ Informasi Debug")
    debug_col1, debug_col2 = st.columns(2)
    with debug_col1:
        st.markdown("**Status Sistem:**")
        st.json({
            "Resolusi Aktif": f"{res_width}x{res_height}",
            "Total Deteksi": len(st.session_state.detections),
            "FPS": cap.get(cv2.CAP_PROP_FPS) if cap.isOpened() else 0
        })
    with debug_col2:
        st.markdown("**Konfigurasi Kamera:**")
        st.json({
            "Frame Width": cap.get(cv2.CAP_PROP_FRAME_WIDTH) if cap.isOpened() else 0,
            "Frame Height": cap.get(cv2.CAP_PROP_FRAME_HEIGHT) if cap.isOpened() else 0
        })