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

# ==============================
# KONFIGURASI UTAMA
# ==============================
MODEL_PATH = "./Model_best/plat_nomor_best.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())

# ==============================
# FUNGSI UTAMA
# ==============================
def main():
   
    
    # Header Section
    with st.container():
        col1, col2 = st.columns([1, 3])
        with col1:
            st.image("https://cdn-icons-png.flaticon.com/512/2331/2331945.png", width=100)
        with col2:
            st.markdown('<h1 class="title-text">AUTOMATIC NUMBER PLATE RECOGNITION SYSTEM</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ SYSTEM SETTINGS")
        with st.expander("VIDEO CONFIG", expanded=True):
            res_options = {
                'HD (1280x720)': (1280, 720),
                'Full HD (1920x1080)': (1920, 1080),
                '480p (640x480)': (640, 480),
                'Custom': 'custom'
            }
            selected_res = st.selectbox("Resolution", list(res_options.keys()))
            
            if selected_res == 'Custom':
                custom_res = st.text_input("Custom Resolution (widthxheight)", "640x480")
                try:
                    res_width, res_height = map(int, custom_res.split('x'))
                except:
                    st.warning("Invalid format! Using default 640x480")
                    res_width, res_height = 640, 480
            else:
                res_width, res_height = res_options[selected_res]
            
        with st.expander("AI SETTINGS"):
            conf_threshold = st.slider("Detection Confidence", 1, 100, 45) / 100
            ocr_confidence = st.slider("OCR Confidence", 1, 100, 30) / 100
            show_debug = st.checkbox("Show Debug Info")

    
    # Main Content
    col1, col2 = st.columns([3, 1])
    
    
    
    # Model Loading
    try:
        model = YOLO(MODEL_PATH).to(DEVICE)
    except Exception as e:
        st.error(f"Model loading failed: {str(e)}")
        st.stop()
    
    # Video Processing
    if 'detections' not in st.session_state:
        st.session_state.detections = []
    
    PLATE_REGEX = re.compile(
        r'^[A-Z]{1,2}\s?\d{1,4}\s?[A-Z]{1,3}$|^[A-Z]{1,2}\d{1,4}[A-Z]{1,3}$'
    )
    def validate_plate(text):
        clean_text = ''.join(filter(str.isalnum, text)).upper()
        date_pattern = r'(\d{2}\.?\d{2}|\d{4,})'
        if re.fullmatch(date_pattern, clean_text):
            return None
        if PLATE_REGEX.match(clean_text):
            parts = re.split(r'(\d+)', clean_text)
            if len(parts) >= 3:
                return f"{parts[0]} {parts[1]} {''.join(parts[2:])}"
            return clean_text
        return None
    
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
                    ocr_result = reader.readtext(plate_roi, detail=0, text_threshold=ocr_confidence)
                    if ocr_result:
                        validated_plate = None
                        for text in ocr_result:
                            plate = validate_plate(text)
                            if plate:
                                validated_plate = plate
                                break
                        if not validated_plate:
                            combined_text = ' '.join(ocr_result)
                            validated_plate = validate_plate(combined_text)
                        if validated_plate:
                            cv2.putText(frame, validated_plate, (x1, y1-10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
                            detection_data = {
                                'ID': str(uuid.uuid4())[:6],
                                'Waktu': datetime.now().strftime("%H:%M:%S"),
                                'Plat': validated_plate,
                                'Kepercayaan Deteksi': f"{conf:.2%}",
                                'Resolusi': f"{res_width}x{res_height}",
                                'Posisi': f"({x1},{y1}) - ({x2},{y2})"
                            }
                            current_detections.append(detection_data)
                except Exception as e:
                    if show_debug:
                        st.error(f"Error processing ROI: {str(e)}")
        
        st.session_state.detections.extend(current_detections)
        return frame

    col1, col2 = st.columns([3, 1])
    with col1:
        video_source = st.radio("Pilih Sumber Video:", 
                            ("Webcam", "Upload Video"), 
                            horizontal=True)

    with col2:
        if st.button("🔄 Reset Deteksi"):
            st.session_state.detections = []
            st.experimental_rerun()

    video_placeholder = st.empty()
    data_placeholder = st.empty()

    if video_source == "Webcam":
        cap = cv2.VideoCapture(4)
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
        video_placeholder.image(processed_frame[:, :, ::-1], 
                            channels="RGB", 
                            use_container_width=True)
        
        if st.session_state.detections:
            df = pd.DataFrame(st.session_state.detections).drop_duplicates('Plat', keep='last')
            data_placeholder.dataframe(
                df[['ID', 'Waktu', 'Plat', 'Kepercayaan Deteksi', 'Resolusi']],
                height=300,
                use_container_width=True
            )

    cap.release()

    # ... (Rest of your existing processing functions here) ...

    # ==============================
    # TAMPILKAN DETECTIONS
    # ==============================
    if st.session_state.detections:
        df = pd.DataFrame(st.session_state.detections).drop_duplicates('Plat', keep='last')
        with data_placeholder.container():
            st.markdown(f"🔍 **Live Detections** <span class='badge'>{len(df)}</span>", unsafe_allow_html=True)
            
            for _, row in df.iterrows():
                with st.container():
                    st.markdown(f"""
                        <div style="background: rgba(255,255,255,0.1); 
                                    padding: 1rem; 
                                    border-radius: 10px;
                                    margin: 0.5rem 0;
                                    box-shadow: 0 2px 4px rgba(0,0,0,0.1)">
                            <div style="font-size: 1.1rem; font-weight: bold; color: #00b4d8">
                                🚘 {row['Plat']}
                            </div>
                            <div style="display: flex; justify-content: space-between; margin-top: 0.5rem">
                                <div>🕒 {row['Waktu']}</div>
                                <div>🔮 {row['Kepercayaan Deteksi']}</div>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
            
            # Download Button
            csv = df.to_csv(index=False).encode()
            st.download_button(
                label="📥 Export as CSV",
                data=csv,
                file_name="detections.csv",
                mime="text/csv",
                use_container_width=True
            )

if __name__ == "__main__":
    main()