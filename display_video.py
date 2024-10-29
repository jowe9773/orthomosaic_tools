import cv2
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

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

        # Bind resizing event
        self.root.bind("<Configure>", self.on_resize)

        # Start video playback loop
        self.update_video()

    def open_video(self):
        # Open file dialog to select video
        self.video_path = filedialog.askopenfilename(title="Select Video File", filetypes=[("MP4 files", "*.mp4"), ("All files", "*.*")])
        if self.video_path:
            self.cap = cv2.VideoCapture(self.video_path)
            if not self.cap.isOpened():
                messagebox.showerror("Error", "Could not open video.")
                return
            
            # Get frame rate and reset frame position
            self.frame_rate = self.cap.get(cv2.CAP_PROP_FPS) or 30
            self.current_frame = 0
            self.paused = False  # Ensure video starts unpaused

    def update_speed(self, value):
        # Update playback speed based on slider value
        self.playback_speed = float(value)

    def skip(self, seconds):
        # Skip frames forward or backward by a specified number of seconds
        if self.cap and self.cap.isOpened():
            self.current_frame += int(seconds * self.frame_rate)
            self.current_frame = max(0, self.current_frame)
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)

    def toggle_play_pause(self):
        # Toggle play/pause state
        self.paused = not self.paused
        self.play_pause_button.config(text="Play" if self.paused else "Pause")

    def on_resize(self, event):
        # Adjust display dimensions to match the resized window, keeping space for control panel
        if event.widget == self.root:
            new_width = max(event.width - 200, 1)
            new_height = max(event.height, 1)
            self.display_width, self.display_height = new_width, new_height

    def update_video(self):
        # Display video frames
        if self.cap and self.cap.isOpened() and not self.paused:
            ret, frame = self.cap.read()

            if ret:
                # Update current frame position
                self.current_frame = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))

                # Scale video to match current window dimensions
                scale_factor = min(self.display_width / frame.shape[1], self.display_height / frame.shape[0])
                resized_frame = cv2.resize(frame, (int(frame.shape[1] * scale_factor), int(frame.shape[0] * scale_factor)), interpolation=cv2.INTER_AREA)

                # Convert frame to ImageTk format and display
                image = Image.fromarray(cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB))
                image_tk = ImageTk.PhotoImage(image=image)
                self.display_label.config(image=image_tk)
                self.display_label.image = image_tk  # Keep a reference to avoid garbage collection
            else:
                # Loop video back to the start if at the end
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        # Schedule next frame update with adjusted speed
        self.root.after(int(1000 / (self.frame_rate * self.playback_speed)), self.update_video)

# Set up main application window
root = tk.Tk()
app = VideoPlayer(root)
root.mainloop()
