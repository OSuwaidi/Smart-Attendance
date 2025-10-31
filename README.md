# 📋 Smart Attendance System

A **Face Recognition-based Smart Attendance System** built with **Streamlit**, **OpenCV**, and **streamlit-webrtc**.  

This application allows you to:  
- 👤 **Register faces** using your webcam (stores multiple images per person).  
- ✅ **Take attendance** by recognizing registered faces in real time.  
- 📅 **View daily attendance records** stored as CSV files.  

---

## 🚀 Features
- Real-time video processing with **streamlit-webrtc**  
- Face detection using **Haar Cascade Classifier**  
- Face recognition with **KNN classifier**  
- Attendance stored in `Attendance/Attendance_<date>.csv`  
- Simple, interactive **Streamlit UI**  

---

## 🛠️ Installation

### 1. Clone the Repository
```
git clone https://github.com/<your-username>/smart-attendance-system.git
```
```
cd smart-attendance-system
```

### 2. Create Virtual Environment (Recommended)
```
conda create -n smart_attendance python=3.11
```
```
conda activate smart_attendance
```

### 3. Install Dependencies
```
pip install -r requirements.txt
```


## ▶️ Usage

### 1. Run the App
```
streamlit run app.py
```

### 2. Register New Face
- Enter your **name**  
- Click **Start Registration**
- Click the **Start** button to turn on the camera
- The system will capture 5 images of your face  

### 3. Take Attendance
- Start the camera under **Take Attendance**  
- The app will recognize registered faces  
- Click **Mark Attendance** to record it  

### 4. View Today’s Attendance
- The latest attendance is shown in the **📅 Today's Attendance** section  
- Attendance is stored as: CSV file 
