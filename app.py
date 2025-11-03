import streamlit as st
import cv2
import pandas as pd
import os
from datetime import datetime
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode
import time
import numpy as np
import threading
import av  # Needed to return frames correctly
import warnings
import torch
from torchvision.transforms import v2
from zoneinfo import ZoneInfo
from face_registration import save_face_data
from take_attendance import mark_attendance
from ... import detector, encoder

warnings.filterwarnings("ignore")

# --- Page Configuration ---
st.set_page_config(page_title="Smart Attendance System", layout="centered")
st.title("📋 Smart Attendance System")

# --- Initialize Session State ---
if "init" not in st.session_state:
    st.session_state.start_registration = False
    st.session_state.device = "cuda"
    st.session_state.detector = detector.to("cuda").eval()
    st.session_state.encoder = encoder.to("cuda").eval()


def pre_process(img: np.ndarray) -> torch.Tensor:
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    T = v2.Compose([v2.ToImage(), v2.Resize((112, 112)), v2.ToDtype(torch.float32, scale=True), v2.Normalize(
            mean=[0.302765429019928, 0.2525686025619507, 0.21395820379257202],
            std=[0.1909194439649582, 0.1914139688014984, 0.19042560458183289]
        )])
    return T(img).unsqueeze().to(st.session_state.device)  # (1, 3, H, W)


# --- Video Processor for Registration ---
class RegistrationProcessor(VideoTransformerBase):
    def __init__(self):
        self.lock = threading.Lock()
        self.last_capture_time = 0
        self.frame_count = 0
        self.local_captures = []

    def recv(self, frame):
        self.frame_count += 1
        img = frame.to_ndarray(format="bgr24")

        if len(self.local_captures) < 5:
            with torch.inference_mode():
                face_bboxes = st.session_state.detector(img)

            with self.lock:
                current_time = time.time()
                if current_time - self.last_capture_time > 1.5:  # can we put this condition before detector?
                    if len(face_bboxes):
                        st.session_state.feedback = "Face Detected!"
                        x1, y1, x2, y2 = map(int, face_bboxes[0])
                        crop_face = img[y1:y2, x1:x2]
                        if crop_face.size:
                            crop_face_tensor = pre_process(crop_face)
                            with torch.inference_mode():
                                face_embedding = st.session_state.encoder(crop_face_tensor)
                            self.local_captures.append(face_embedding)
                            self.last_capture_time = current_time
                            print(f"*** CAPTURED IMAGE #{self.local_captures} ***")
                    else:
                        st.session_state.feedback = "No Face Detected"

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- App Sections ---

# Section 1: Register New Face
with st.container():
    st.subheader("🧑‍💻 Register New Face")
    st.session_state.new_name = st.text_input("Enter your name:", value=st.session_state.get('new_name', ''), key="new_name_input")

    if st.button("📸 Start Registration", key="start_reg_btn"):
        if not st.session_state.new_name:
            st.warning("Please enter a name before registering.")
        else:
            st.session_state.captured_embeddings = []
            st.session_state.start_registration = True
            st.rerun()

    if st.session_state.start_registration:
        if len(st.session_state.captured_embeddings) < 5:
            st.warning("Click the 'START' button below to turn on your camera.")

            ctx = webrtc_streamer(
                key="registration",
                mode=WebRtcMode.SENDRECV,
                video_processor_factory=RegistrationProcessor,
                media_stream_constraints={"video": True, "audio": False},
                async_processing=False
            )

            if ctx.video_processor:
                with ctx.video_processor.lock:
                    st.session_state.captured_embeddings = ctx.video_processor.local_captures.copy()

            if ctx.state.playing:
                st.info("Please show your face to the camera. Capturing 5 images...")

            st.info(st.session_state.get('feedback', 'Initializing...'))
            st.progress(len(st.session_state.captured_embeddings) / 5)
            st.write(f"Captured: {len(st.session_state.captured_embeddings)}/5")

            if ctx.state.playing:
                time.sleep(0.2)
                st.rerun()

        else:
            st.session_state.start_registration = False
            st.success("Capture complete! Saving your face data...")
            st.balloons()

            success, message = save_face_data(st.session_state.new_name, st.session_state.captured_embeddings)
            if success:
                st.success(message)
                st.info("Data saved. Please refresh the page to update the attendance model.")
            else:
                st.error(message)

            st.session_state.captured_embeddings = []
            st.session_state.new_name = ""
            time.sleep(3)
            st.rerun()

# Section 2: Take Attendance
with st.container():
    st.subheader("✅ Take Attendance")
    attendance_register=set()
    st.info("Click 'START' below to begin attendance.")

    class AttendanceProcessor(VideoTransformerBase):
        def recv(self, frame):
            recognized_name = ""
            img = frame.to_ndarray(format="bgr24")
            img = cv2.resize(img, (112, 112))  # 112, 112, 3
            img_tensor = torch.Tensor(img).view(1, 3, 112, 112).to(device)
            backbone = backbone.to(device)
            with torch.inference_mode():
                embedding, bbox = backbone(img_tensor)
            embedding, (x1, y1, x2, y2) = embedding.T.cpu(), bbox.cpu()

            crop_img = img[y1:y2, x1:x2]
            if crop_img.size > 0:
                gallary = load_gallary()  # (N, D)
                distances = gallary @ embedding  # (N, 1)
                matching_index = distances.argmin(0)
                recognized_name = names[matching_index]
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, recognized_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (255, 255, 255), 2)

            st.session_state["recognized_name"] = recognized_name
            if recognized_name and recognized_name not in attendance_register:
                attendance_register.add(recognized_name)
                message = mark_attendance(recognized_name)
                st.info(message)  # Shows success/error in the UI
                print(f"Message is {message}")
                if "Error" in message:
                    st.error(message)
                else:
                    st.success(message)
            return av.VideoFrame.from_ndarray(img, format="bgr24")

    ctx_att = webrtc_streamer(
        key="attendance",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=AttendanceProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=False
    )

    if ctx_att.state.playing:
        time.sleep(0.1)
        st.rerun()

# Section 3: Show Today’s Attendance
with st.container():
    st.subheader("📅 Today's Attendance")
    date_str = datetime.now(ZoneInfo("Asia/Kolkata")).strftime("%d-%m-%Y")
    filename = f"Attendance/Attendance_{date_str}.csv"
    if os.path.exists(filename):
        try:
            df = pd.read_csv(filename)
            st.dataframe(df)
        except Exception as e:
            st.error(f"Could not read the attendance file: {e}")
    else:
        st.warning("No attendance has been recorded for today yet.")
