import cv2
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np

class VideoPlayer:
    def __init__(self, root):
        self.root = root
        self.root.title("Video Player")

        # Initial dimensions for video display
        self.display_width = 960
        self.display_height = 720

        # Set initial window size
        self.root.geometry(f"{self.display_width + 200}x{self.display_height}")
        self.root.minsize(800, 600)

        # Video player attributes
        self.video_path = None
        self.cap = None
        self.frame_rate = 30  # Default frame rate
        self.paused = False
        self.current_frame = 0
        self.playback_speed = 1.0  # Default playback speed multiplier

        # Video display label
        self.display_label = tk.Label(root, bg="black")
        self.display_label.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

        # Configure the grid layout to expand with window resizing
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)

        # Control panel on the right
        self.control_panel = tk.Frame(root)
        self.control_panel.grid(row=0, column=1, sticky="ns", padx=10, pady=10)

        # Open video button
        open_button = tk.Button(self.control_panel, text="Open Video", command=self.open_video)
        open_button.pack(pady=10)

        # Play/Pause button
        self.play_pause_button = tk.Button(self.control_panel, text="Pause", command=self.toggle_play_pause)
        self.play_pause_button.pack(pady=10)

        # Playback speed slider
        self.speed_slider = tk.Scale(self.control_panel, from_=0.1, to=20.0, resolution=0.1, orient=tk.HORIZONTAL, label="Playback Speed", command=self.update_speed)
        self.speed_slider.set(1.0)  # Default speed
        self.speed_slider.pack(pady=10)

        # Skip control buttons
        tk.Button(self.control_panel, text="<< 10 sec", command=lambda: self.skip(-10)).pack(pady=5)
        tk.Button(self.control_panel, text="<< 1 sec", command=lambda: self.skip(-1)).pack(pady=5)
        tk.Button(self.control_panel, text=">> 1 sec", command=lambda: self.skip(1)).pack(pady=5)
        tk.Button(self.control_panel, text=">> 10 sec", command=lambda: self.skip(10)).pack(pady=5)

        # Time entry box and total duration label
        self.time_entry = tk.Entry(self.control_panel, width=10, justify="center")
        self.time_entry.bind("<Return>", self.jump_to_time)
        self.time_entry.pack(pady=5)
        self.total_time_label = tk.Label(self.control_panel, text="Total: 00:00:00")
        self.total_time_label.pack()

        # Bind resizing event
        self.root.bind("<Configure>", self.on_resize)

        # Start video playback loop
        self.update_video()

    def open_video(self):
        self.video_path = filedialog.askopenfilename(title="Select Video File", filetypes=[("MP4 files", "*.mp4"), ("All files", "*.*")])
        if self.video_path:
            self.cap = cv2.VideoCapture(self.video_path)
            if not self.cap.isOpened():
                messagebox.showerror("Error", "Could not open video.")
                return
            
            # Get frame rate and reset frame position
            self.frame_rate = self.cap.get(cv2.CAP_PROP_FPS) or 30
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.total_seconds = self.total_frames / self.frame_rate

            # Update total time label
            total_time_str = self.format_time(self.total_seconds)
            self.total_time_label.config(text=f"Total: {total_time_str}")

            # Reset the time entry
            self.time_entry.delete(0, tk.END)
            self.time_entry.insert(0, "00:00:00")

            # Start video playback
            self.current_frame = 0
            self.paused = False

    def format_time(self, seconds):
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        seconds = int(seconds % 60)
        return f"{hours:02}:{minutes:02}:{seconds:02}"

    def jump_to_time(self, event=None):
        try:
            time_str = self.time_entry.get()
            hours, minutes, seconds = map(int, time_str.split(":"))
            target_seconds = hours * 3600 + minutes * 60 + seconds
            target_frame = int(target_seconds * self.frame_rate)
            target_frame = min(max(0, target_frame), self.total_frames - 1)  # Bound the frame within the total frames

            # Set the video to the target frame
            self.current_frame = target_frame
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
        except ValueError:
            messagebox.showerror("Invalid Time Format", "Please enter time in HH:MM:SS format.")

    def update_speed(self, value):
        self.playback_speed = float(value)

    def skip(self, seconds):
        if self.cap and self.cap.isOpened():
            self.current_frame += int(seconds * self.frame_rate)
            self.current_frame = max(0, min(self.current_frame, self.total_frames - 1))
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)

    def toggle_play_pause(self):
        self.paused = not self.paused
        self.play_pause_button.config(text="Play" if self.paused else "Pause")

    def on_resize(self, event):
        if event.widget == self.root:
            new_width = max(event.width - 200, 1)
            new_height = max(event.height, 1)
            self.display_width, self.display_height = new_width, new_height

    def update_video(self):
        if self.cap and self.cap.isOpened() and not self.paused:
            ret, frame = self.cap.read()

            if ret:
                self.current_frame = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
                self.time_entry.delete(0, tk.END)
                self.time_entry.insert(0, self.format_time(self.current_frame / self.frame_rate))

                # Resize and display frame
                scale_factor = min(self.display_width / frame.shape[1], self.display_height / frame.shape[0])
                resized_frame = cv2.resize(frame, (int(frame.shape[1] * scale_factor), int(frame.shape[0] * scale_factor)), interpolation=cv2.INTER_AREA)

                image = Image.fromarray(cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB))
                image_tk = ImageTk.PhotoImage(image=image)
                self.display_label.config(image=image_tk)
                self.display_label.image = image_tk
            else:
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        self.root.after(int(1000 / (self.frame_rate * self.playback_speed)), self.update_video)

root = tk.Tk()
app = VideoPlayer(root)
root.mainloop()
