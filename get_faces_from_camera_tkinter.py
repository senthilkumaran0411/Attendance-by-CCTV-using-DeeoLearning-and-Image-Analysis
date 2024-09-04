import os
import time
import cv2
import numpy as np
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import font as tkFont

class Face_Register:
    def __init__(self, window):
        self.win = window
        self.win.title("Face Register")
        self.win.geometry("1000x500")

        self.current_frame_faces_cnt = 0
        self.existing_faces_cnt = 0
        self.ss_cnt = 0
        self.fps = 0
        self.start_time = time.time()

        # Initialize face detector
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

        # Tkinter GUI elements
        self.frame_left_camera = tk.Frame(self.win)
        self.label = tk.Label(self.win)
        self.label.pack(side=tk.LEFT)
        self.frame_left_camera.pack()

        self.frame_right_info = tk.Frame(self.win)
        self.label_cnt_face_in_database = tk.Label(self.frame_right_info, text=str(self.existing_faces_cnt))
        self.label_fps_info = tk.Label(self.frame_right_info, text="")
        self.input_name = tk.Entry(self.frame_right_info)
        self.input_name_char = ""
        self.label_warning = tk.Label(self.frame_right_info)
        self.label_face_cnt = tk.Label(self.frame_right_info, text="Faces in current frame: ")
        self.log_all = tk.Label(self.frame_right_info)

        self.font_title = tkFont.Font(family='Helvetica', size=20, weight='bold')
        self.font_step_title = tkFont.Font(family='Helvetica', size=15, weight='bold')
        self.font_warning = tkFont.Font(family='Helvetica', size=15, weight='bold')

        self.path_photos_from_camera = "data/data_faces_from_camera/"
        self.current_face_dir = ""
        self.font = cv2.FONT_ITALIC

        # Camera selection
        self.camera_index = tk.IntVar()
        self.camera_index.set(0)

        self.frame_camera_select = tk.Frame(self.win)
        self.label_camera_select = tk.Label(self.frame_camera_select, text="Select Camera:")
        self.label_camera_select.pack(side=tk.LEFT)

        self.camera_dropdown = tk.OptionMenu(self.frame_camera_select, self.camera_index, *self.get_camera_options())
        self.camera_dropdown.pack(side=tk.LEFT)

        self.frame_camera_select.pack(pady=10)

        self.cap = cv2.VideoCapture(self.camera_index.get())

        # Update camera when dropdown changes
        self.camera_index.trace_add('write', self.update_camera)

        self.GUI_info()

    def get_camera_options(self):
        cameras = self.get_available_cameras()
        return [f"Camera {i}" for i in cameras]

    def get_available_cameras(self):
        available_cameras = []
        for index in range(10):  # Checking first 10 camera indices, adjust as needed
            cap = cv2.VideoCapture(index)
            if cap.isOpened():
                available_cameras.append(index)
                cap.release()
        return available_cameras

    def update_camera(self, *args):
        self.cap.release()
        self.cap = cv2.VideoCapture(self.camera_index.get())

    def GUI_clear_data(self):
        folders_rd = os.listdir(self.path_photos_from_camera)
        for folder in folders_rd:
            folder_path = os.path.join(self.path_photos_from_camera, folder)
            import shutil
            shutil.rmtree(folder_path)
        if os.path.isfile("data/features_all.csv"):
            os.remove("data/features_all.csv")
        self.label_cnt_face_in_database['text'] = "0"
        self.existing_faces_cnt = 0
        self.log_all["text"] = "Face images and `features_all.csv` removed!"

    def GUI_get_input_name(self):
        self.input_name_char = self.input_name.get()
        self.create_face_folder()
        self.label_cnt_face_in_database['text'] = str(self.existing_faces_cnt)

    def GUI_info(self):
        tk.Label(self.frame_right_info,
                 text="Face register",
                 font=self.font_title).grid(row=0, column=0, columnspan=3, sticky=tk.W, padx=2, pady=20)

        tk.Label(self.frame_right_info, text="FPS: ").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        self.label_fps_info.grid(row=1, column=1, sticky=tk.W, padx=5, pady=2)

        tk.Label(self.frame_right_info, text="Faces in database: ").grid(row=2, column=0, sticky=tk.W, padx=5, pady=2)
        self.label_cnt_face_in_database.grid(row=2, column=1, sticky=tk.W, padx=5, pady=2)

        tk.Label(self.frame_right_info,
                 text="Faces in current frame: ").grid(row=3, column=0, columnspan=2, sticky=tk.W, padx=5, pady=2)
        self.label_face_cnt.grid(row=3, column=2, columnspan=3, sticky=tk.W, padx=5, pady=2)

        self.label_warning.grid(row=4, column=0, columnspan=3, sticky=tk.W, padx=5, pady=2)

        # Step 1: Clear old data
        tk.Label(self.frame_right_info,
                 font=self.font_step_title,
                 text="Step 1: Clear face photos").grid(row=5, column=0, columnspan=2, sticky=tk.W, padx=5, pady=20)
        tk.Button(self.frame_right_info,
                  text='Clear',
                  command=self.GUI_clear_data).grid(row=6, column=0, columnspan=3, sticky=tk.W, padx=5, pady=2)

        # Step 2: Input name and create folders for face
        tk.Label(self.frame_right_info,
                 font=self.font_step_title,
                 text="Step 2: Input name").grid(row=7, column=0, columnspan=2, sticky=tk.W, padx=5, pady=20)

        tk.Label(self.frame_right_info, text="Name: ").grid(row=8, column=0, sticky=tk.W, padx=5, pady=0)
        self.input_name.grid(row=8, column=1, sticky=tk.W, padx=0, pady=2)

        tk.Button(self.frame_right_info,
                  text='Input',
                  command=self.GUI_get_input_name).grid(row=8, column=2, padx=5)

        # Step 3: Save current face in frame
        tk.Label(self.frame_right_info,
                 font=self.font_step_title,
                 text="Step 3: Save face image").grid(row=9, column=0, columnspan=2, sticky=tk.W, padx=5, pady=20)

        tk.Button(self.frame_right_info,
                  text='Save current face',
                  command=self.save_current_face).grid(row=10, column=0, columnspan=3, sticky=tk.W)

        # Show log in GUI
        self.log_all.grid(row=11, column=0, columnspan=20, sticky=tk.W, padx=5, pady=20)

        self.frame_right_info.pack()

    def pre_work_mkdir(self):
        if not os.path.exists(self.path_photos_from_camera):
            os.makedirs(self.path_photos_from_camera, exist_ok=True)

    def check_existing_faces_cnt(self):
        if os.listdir("data/data_faces_from_camera/"):
            # Get the order of the latest person
            person_list = os.listdir("data/data_faces_from_camera/")
            person_num_list = []
            for person in person_list:
                try:
                    person_order = person.split('_')[1].split('_')[0]
                    person_num_list.append(int(person_order))
                except (IndexError, ValueError):
                    continue
            self.existing_faces_cnt = max(person_num_list, default=0)
        else:
            self.existing_faces_cnt = 0

    def update_fps(self):
        now = time.time()
        if int(self.start_time) != int(now):
            self.fps_show = self.fps
            self.fps = 0
        else:
            self.fps += 1
        self.start_time = now
        self.label_fps_info.config(text=str(self.fps_show))

    def create_face_folder(self):
        folder_name = f"person_{self.existing_faces_cnt + 1}"
        self.current_face_dir = os.path.join(self.path_photos_from_camera, folder_name)
        os.makedirs(self.current_face_dir, exist_ok=True)

    def save_current_face(self):
        if self.current_frame_faces_cnt > 0:
            img_path = os.path.join(self.current_face_dir, f"{self.ss_cnt}.jpg")
            cv2.imwrite(img_path, self.current_frame)
            self.ss_cnt += 1
            self.log_all["text"] = f"Face image saved: {img_path}"
        else:
            self.log_all["text"] = "No face detected in current frame"

    def process(self):
        self.start_time = time.time()
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break
            self.current_frame = frame
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
            self.current_frame_faces_cnt = len(faces)
            self.label_face_cnt['text'] = f"Faces in current frame: {self.current_frame_faces_cnt}"
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            img = Image.fromarray(frame)
            img = ImageTk.PhotoImage(img)
            self.label.config(image=img)
            self.label.image = img
            self.update_fps()
            self.win.update_idletasks()
            self.win.update()
        self.cap.release()

    def run(self):
        self.pre_work_mkdir()
        self.check_existing_faces_cnt()
        self.GUI_info()
        self.process()

    def __del__(self):
        self.cap.release()

if __name__ == "__main__":
    root = tk.Tk()
    app = Face_Register(root)
    app.run()
