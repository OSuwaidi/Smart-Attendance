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
from zoneinfo import ZoneInfo

warnings.filterwarnings("ignore")

# Import functions from our custom modules
from face_registration import save_face_data
from take_attendance import load_model, mark_attendance

# --- Page Configuration ---
st.set_page_config(page_title="Smart Attendance System", layout="centered")
st.title("📋 Smart Attendance System")

# --- Initialize Session State ---
if "start_registration" not in st.session_state:
    st.session_state.start_registration = False
if "captured_faces" not in st.session_state:
    st.session_state.captured_faces = []
if "new_name" not in st.session_state:
    st.session_state.new_name = ""
if "feedback" not in st.session_state:
    st.session_state.feedback = ""
if "recognized_name" not in st.session_state:
    st.session_state.recognized_name = "Unknown"

# --- Load Models and Data ---
try:
    facedetect = cv2.CascadeClassifier('Data/haarcascade_frontalface_default.xml')
except Exception as e:
    st.error(f"Error loading Haar Cascade file: {e}. Make sure the file is in the 'Data' directory.")
    st.stop()

knn, error_message = load_model()
if error_message:
    st.warning(error_message)

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

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = facedetect.detectMultiScale(gray, 1.3, 5)

        with self.lock:
            current_time = time.time()
            if len(faces) > 0:
                st.session_state.feedback = "Face Detected!"
                if current_time - self.last_capture_time > 1.0:
                    if len(self.local_captures) < 5:
                        (x, y, w, h) = faces[0]
                        crop_img = img[y:y+h, x:x+w]
                        if crop_img.size > 0:
                            resized_img = cv2.resize(crop_img, (50, 50))
                            self.local_captures.append(resized_img)
                            self.last_capture_time = current_time
                            print(f"*** CAPTURED IMAGE #{len(self.local_captures)} ***")
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
            st.session_state.captured_faces = []
            st.session_state.start_registration = True
            st.rerun()

    if st.session_state.start_registration:
        if len(st.session_state.captured_faces) < 5:
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
                    st.session_state.captured_faces = ctx.video_processor.local_captures.copy()

            if ctx.state.playing:
                st.info("Please show your face to the camera. Capturing 5 images...")

            st.info(st.session_state.get('feedback', 'Initializing...'))
            st.progress(len(st.session_state.captured_faces) / 5)
            st.write(f"Captured: {len(st.session_state.captured_faces)}/5")

            if ctx.state.playing:
                time.sleep(0.2)
                st.rerun()

        else:
            st.session_state.start_registration = False
            st.success("Capture complete! Saving your face data...")
            st.balloons()

            success, message = save_face_data(st.session_state.new_name, st.session_state.captured_faces)
            if success:
                st.success(message)
                st.info("Data saved. Please refresh the page to update the attendance model.")
            else:
                st.error(message)

            st.session_state.captured_faces = []
            st.session_state.new_name = ""
            time.sleep(3)
            st.rerun()

# Section 2: Take Attendance
with st.container():
    st.subheader("✅ Take Attendance")
    attendance_register=set()
    if knn:
        st.info("Click 'START' below to begin attendance.")

        class AttendanceProcessor(VideoTransformerBase):
            def recv(self, frame):
                # what is frame dtype?
                recognized_name = ""
                img = frame.to_ndarray(format="bgr24")
                rgb_face = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                rgb_face = cv2.resize(rgb_face, (112, 112))  # 112, 112, 3
                rgb_tensor = torch.Tensor(rgb_face).view(1, 3, 112, 112).to(device)
                backbone = backbone.to(device)
                with torch.inference_mode():
                    embedding, bbox = backbone(rgb_tensor)
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
    else:
        st.info("Please register a face before taking attendance.")

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
                faces = facedetect.detectMultiScale(gray, 1.3, 5)

                for (x, y, w, h) in faces:
                    crop_img = img[y:y+h, x:x+w]
                    if crop_img.size > 0:
                        resized_img = cv2.resize(crop_img, (50, 50)).flatten().reshape(1, -1)
                        output = knn.predict(resized_img)
                        recognized_name = output[0]
                        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
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
    else:
        st.info("Please register a face before taking attendance.")

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