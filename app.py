import cv2
import numpy as np
import streamlit as st
import av
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration

st.set_page_config(page_title="Motion Detector", layout="wide")
st.title("🎥 Real-Time Motion Detector")
st.markdown("Motion detection via your browser's webcam using OpenCV.")

# Sidebar controls
st.sidebar.header("⚙️ Settings")
threshold    = st.sidebar.slider("Motion Sensitivity", 5, 50, 25)
blur_size    = st.sidebar.slider("Blur Kernel Size (odd only)", 3, 21, 5, step=2)
min_area     = st.sidebar.slider("Min Contour Area", 100, 5000, 500)
show_contour = st.sidebar.checkbox("Show Bounding Boxes", value=True)

# WebRTC config — these are public STUN servers, required for cloud
RTC_CONFIG = RTCConfiguration({
    "iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {"urls": ["stun:stun1.l.google.com:19302"]},
        {"urls": ["stun:stun2.l.google.com:19302"]},
        {"urls": ["stun:stun.relay.metered.ca:80"]},
    ]
})

class MotionDetector(VideoProcessorBase):
    def __init__(self):
        self.prev_gray = None

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (blur_size, blur_size), 0)

        if self.prev_gray is None:
            self.prev_gray = gray
            return av.VideoFrame.from_ndarray(img, format="bgr24")

        # Frame differencing
        diff    = cv2.absdiff(self.prev_gray, gray)
        _, thresh = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
        dilated = cv2.dilate(thresh, None, iterations=2)

        contours, _ = cv2.findContours(
            dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        motion_detected = False

        for contour in contours:
            if cv2.contourArea(contour) < min_area:
                continue
            motion_detected = True
            if show_contour:
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(img, "MOTION", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # Status label on frame
        label = "MOTION DETECTED" if motion_detected else "No Motion"
        color = (0, 0, 255)      if motion_detected else (0, 200, 0)
        cv2.putText(img, label, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        self.prev_gray = gray
        return av.VideoFrame.from_ndarray(img, format="bgr24")


webrtc_streamer(
    key="motion-detector",
    video_processor_factory=MotionDetector,
    rtc_configuration=RTC_CONFIG,
    media_stream_constraints={"video": True, "audio": False},
)

st.markdown("---")
st.caption("Built with OpenCV + Streamlit WebRTC")