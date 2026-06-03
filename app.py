import cv2
import numpy as np
import streamlit as st
from PIL import Image
import time

st.set_page_config(page_title="Motion Detector", layout="wide")
st.title("🎥 Real-Time Motion Detector")
st.markdown("Uses OpenCV frame differencing to detect motion from your webcam.")

# Sidebar controls
st.sidebar.header("⚙️ Settings")
threshold    = st.sidebar.slider("Motion Sensitivity", 5, 50, 25,
                help="Lower = more sensitive")
blur_size    = st.sidebar.slider("Blur Kernel Size (odd only)", 3, 21, 5, step=2)
min_area     = st.sidebar.slider("Min Contour Area", 100, 5000, 500,
                help="Ignore tiny movements below this area")
show_contour = st.sidebar.checkbox("Show Contours", value=True)
show_mask    = st.sidebar.checkbox("Show Motion Mask", value=False)

# Layout
col1, col2 = st.columns(2)
frame_window   = col1.empty()
mask_window    = col2.empty() if show_mask else None
status_text    = st.empty()
motion_counter = st.empty()

start = st.button("▶️ Start Camera")
stop  = st.button("⏹️ Stop Camera")

if "running" not in st.session_state:
    st.session_state.running = False

if start:
    st.session_state.running = True
if stop:
    st.session_state.running = False

motion_count = 0

if st.session_state.running:
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        st.error("❌ Could not open webcam. Make sure your camera is connected and not in use.")
        st.session_state.running = False
    else:
        ret, prev_frame = cap.read()
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        prev_gray = cv2.GaussianBlur(prev_gray, (blur_size, blur_size), 0)

        while st.session_state.running:
            ret, frame = cap.read()
            if not ret:
                st.warning("⚠️ Failed to grab frame.")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (blur_size, blur_size), 0)

            # Frame difference
            diff = cv2.absdiff(prev_gray, gray)
            _, thresh = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
            dilated = cv2.dilate(thresh, None, iterations=2)

            contours, _ = cv2.findContours(
                dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            motion_detected = False
            display_frame = frame.copy()

            for contour in contours:
                if cv2.contourArea(contour) < min_area:
                    continue
                motion_detected = True
                motion_count += 1
                if show_contour:
                    x, y, w, h = cv2.boundingRect(contour)
                    cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(display_frame, "MOTION", (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Status overlay
            label = "⚡ MOTION DETECTED" if motion_detected else "✅ No Motion"
            color = (0, 0, 255) if motion_detected else (0, 200, 0)
            cv2.putText(display_frame, label, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

            # Show frames
            rgb_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            frame_window.image(rgb_frame, channels="RGB", use_column_width=True)

            if show_mask and mask_window:
                mask_window.image(dilated, use_column_width=True, clamp=True)

            status_text.markdown(
                f"### {'🔴 Motion Detected!' if motion_detected else '🟢 All Clear'}"
            )
            motion_counter.caption(f"Total motion events: {motion_count}")

            prev_gray = gray
            time.sleep(0.03)  # ~30 fps

        cap.release()
        status_text.info("Camera stopped.")