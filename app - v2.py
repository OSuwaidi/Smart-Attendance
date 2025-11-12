import cv2
from utils.utils import MTCNNAdapter
import torch
from generate_embeddings import recognize_unlabeled_faces_image
from test_model import load_model_last_block, get_argparser

# Import functions from our custom modules
from face_registration import save_face_data
from fine_tune_main import main
import cv2
import torch
import threading
from tkinter import *
from PIL import Image, ImageTk
import os
import time

class FaceRecognitionApp:
    def __init__(self, root, backbone, detector, embedding_path, grayscale=False, threshold=0.3, topk=1):
        self.root = root
        self.root.title("Face Recognition System")
        self.root.geometry("950x750")

        self.backbone = backbone
        self.detector = detector
        self.embedding_path = embedding_path
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.grayscale = grayscale
        self.threshold = threshold
        self.topk = topk

        self.cap = None
        self.running = False

        # === Video display ===
        self.video_label = Label(root)
        self.video_label.pack(pady=20)

        # === Buttons ===
        btn_frame = Frame(root)
        btn_frame.pack(pady=20)

        self.start_btn = Button(btn_frame, text="Start Recognition", command=self.start_recognition,
                                width=15, height=2, bg="#4CAF50", fg="white")
        self.start_btn.grid(row=0, column=0, padx=10)

        self.stop_btn = Button(btn_frame, text="Stop Recognition", command=self.stop_recognition,
                               width=15, height=2, bg="#F44336", fg="white")
        self.stop_btn.grid(row=0, column=1, padx=10)

        self.register_btn = Button(btn_frame, text="Register Face", command=self.register_face,
                                   width=15, height=2, bg="#2196F3", fg="white")
        self.register_btn.grid(row=0, column=2, padx=10)

        self.update_btn = Button(btn_frame, text="Update Database", command=self.update_database,
                                 width=15, height=2, bg="#9C27B0", fg="white")
        self.update_btn.grid(row=0, column=3, padx=10)

        self.quit_btn = Button(btn_frame, text="Quit", command=self.quit_app,
                               width=15, height=2, bg="#555555", fg="white")
        self.quit_btn.grid(row=0, column=4, padx=10)

        # === Status label ===
        self.status_label = Label(root, text="Status: Idle", font=("Arial", 12))
        self.status_label.pack(pady=10)

    # -------------------------
    # Real-time recognition
    # -------------------------
    def start_recognition(self):
        if not self.running:
            self.running = True
            self.status_label.config(text="Status: Running recognition...")
            self.cap = cv2.VideoCapture(0)
            self.update_frame()

    def stop_recognition(self):
        self.running = False
        self.status_label.config(text="Status: Stopped")

    def quit_app(self):
        self.running = False
        if self.cap is not None:
            self.cap.release()
        self.root.destroy()

    def update_frame(self):
        if self.running and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                self.status_label.config(text="Error: Cannot read from camera.")
                self.stop_recognition()
                return

            result = recognize_unlabeled_faces_image(
                backbone=self.backbone,
                gallery_npz=self.embedding_path,
                image_array=frame,
                face_detector=self.detector,
                device=self.device,
                crop_size=(112, 112),
                grayscale=self.grayscale,
                topk=self.topk,
                threshold=self.threshold,
                show=False,
                save_path=None
            )

            if result is not None:
                recognized_names, face_bboxes = result
                if recognized_names is not None and face_bboxes is not None:
                    for name, bbox in zip(recognized_names, face_bboxes):
                        x1, y1, x2, y2 = map(int, bbox)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                        cv2.putText(frame, name, (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

            # Convert and show
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)
            self.root.after(30, self.update_frame)
        else:
            if self.cap is not None:
                self.cap.release()
                self.video_label.config(image='')

    # -------------------------
    # Register a new face
    # -------------------------
    def register_face(self):
        name = simpledialog.askstring("Register Face", "Enter your name:", parent=self.root)
        if not name:
            messagebox.showwarning("Warning", "Name cannot be empty!")
            return

        save_dir = os.path.join("dataset", name)
        os.makedirs(save_dir, exist_ok=True)

        cap = cv2.VideoCapture(0)
        count = 0
        messagebox.showinfo("Info", f"Starting to capture 10 images for '{name}'. Please face the camera.")

        while count < 10:
            ret, frame = cap.read()
            if not ret:
                continue

            cv2.imshow("Registering...", frame)
            filename = os.path.join(save_dir, f"img_{count+1}.jpg")
            cv2.imwrite(filename, frame)
            count += 1
            print(f"Saved {count}/10 images to {filename}")
            time.sleep(0.3)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        messagebox.showinfo("Done", f"Registration complete for '{name}'. Images saved to {save_dir}")
        self.status_label.config(text=f"Status: Registered new user '{name}'")

    # -------------------------
    # Update database
    # -------------------------
    def update_database(self):
        try:
            self.status_label.config(text="Status: Updating database...")
            main()  # <-- your update function
            messagebox.showinfo("Success", "Database updated successfully!")
            self.status_label.config(text="Status: Database updated.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to update database:\n{e}")
            self.status_label.config(text="Status: Update failed.")



def get_backbone(snapshot_path):
    args = get_argparser()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    _backbone, _, _ = load_model_last_block(
            args,
            num_classes=1000,
            snapshot_path=snapshot_path,
            device=device,
        )
    _backbone.eval()
    return _backbone

if __name__ == "__main__":
    
    detector = MTCNNAdapter()
    snapshot_path = r'C:\Users\qhd\Desktop\face\FaceNet-main\FaceNet-main\checkpoints\20251111_150701__model_mobilefacenet__head_arcface__opt_adamw__phase_full'
    embedding_path = r'C:\Users\qhd\Desktop\face\FaceNet-main\FaceNet-main\checkpoints\20251111_150701__model_mobilefacenet__head_arcface__opt_adamw__phase_full\gallary.npz'

    backbone = get_backbone(snapshot_path)
    root = Tk()
    app = FaceRecognitionApp(
        root=root,
        backbone=backbone,
        detector=detector,
        embedding_path=embedding_path,
        grayscale=False,
        threshold=0.3
    )
    root.mainloop()
