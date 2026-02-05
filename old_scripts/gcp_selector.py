#A program that allows you to select ground control points on a video file. 

#gcp_selection.py

"""This file will contain the code that opens a GUI based GCP selection tool."""
import tkinter as tk
from tkinter import ttk, filedialog
import csv
import cv2
from PIL import Image, ImageTk


class TargetWidget(ttk.Frame):
    def __init__(self, master=None, variable=None, value=None, app=None, entry_width=5):
        super().__init__(master)
        self.variable = variable
        self.value = value
        self.app = app  # Store reference to the App instance
        self.entry_width = entry_width
        self.create_widgets()

    def create_widgets(self):
        self.radio_button = ttk.Radiobutton(self, variable=self.variable, value=self.value,
                                            command=self.set_active)
        self.radio_button.grid(row=0, column=0, padx=2, pady=5)

        self.target_label = ttk.Label(self, text="Target:")
        self.target_label.grid(row=0, column=1, padx=2, pady=5)
        self.target_entry = ttk.Entry(self, width=self.entry_width)
        self.target_entry.grid(row=0, column=2, padx=2, pady=5)

        self.xpixels_label = ttk.Label(self, text="X Pixels:")
        self.xpixels_label.grid(row=0, column=3, padx=2, pady=5)
        self.xpixels_entry = ttk.Entry(self, width=self.entry_width)
        self.xpixels_entry.grid(row=0, column=4, padx=2, pady=5)

        self.ypixels_label = ttk.Label(self, text="Y Pixels:")
        self.ypixels_label.grid(row=0, column=5, padx=2, pady=5)
        self.ypixels_entry = ttk.Entry(self, width=self.entry_width)
        self.ypixels_entry.grid(row=0, column=6, padx=2, pady=5)

    def set_active(self):
        if self.app.radio_variable.get() == self.value:
            self.app.set_active(self.value)  # Call set_active on the App instance
        self.app.image_viewer.draw_points()  # Redraw points when the active target changes

    def update_coordinates(self, x, y):
        self.xpixels_entry.delete(0, tk.END)
        self.xpixels_entry.insert(0, str(x))
        self.ypixels_entry.delete(0, tk.END)
        self.ypixels_entry.insert(0, str(y))

    def get_coordinates(self):
        try:
            x = int(self.xpixels_entry.get())
            y = int(self.ypixels_entry.get())
            return (x, y)
        except ValueError:
            return None

    def get_target_data(self):
        target = self.target_entry.get()
        coordinates = self.get_coordinates()
        if coordinates:
            x, y = coordinates
            return [target, x, y]
        return [target, "", ""]


class ImageViewer(tk.Frame):
    def __init__(self, master=None, app=None):
        super().__init__(master)
        self.app = app
        self.pack(fill=tk.BOTH, expand=True)
        self.create_widgets()

        self.canvas.bind("<Double-1>", self.on_double_click)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonPress-1>", self.on_drag_start)
        self.canvas.bind("<MouseWheel>", self.on_zoom)

        self.image = None
        self.tk_img = None
        self.scale = 1.0
        self.canvas_origin = (0, 0)
        self.img_pos = (0, 0)  # Top-left corner of the image on the canvas
        self.active_target = None  # Track the active target
        self.points = {}  # Store points for each target widget

    def create_widgets(self):
        self.canvas = tk.Canvas(self, bg="gray")
        self.canvas.pack(fill=tk.BOTH, expand=True)

    def open_image(self):
        filepath = filedialog.askopenfilename()
        if not filepath:
            return

        if ".jpg" in filepath:
            self.image = Image.open(filepath)

        if ".MP4" in filepath:
            cap = cv2.VideoCapture(filepath)
            cap.set(cv2.CAP_PROP_POS_FRAMES, 300)
            ret, frame = cap.read()
            cap.release()
            if ret:
                self.image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            else:
                print("Frame couldn't be opened")

        self.img_pos = (0, 0)
        self.fit_image_to_window()
        self.update_image()

    def fit_image_to_window(self):
        if self.image:
            img_width, img_height = self.image.size
            canvas_width = self.canvas.winfo_width()
            canvas_height = self.canvas.winfo_height()

            scale_x = canvas_width / img_width
            scale_y = canvas_height / img_height
            self.scale = min(scale_x, scale_y)

    def update_image(self):
        if self.image:
            width, height = self.image.size
            scaled_width, scaled_height = int(width * self.scale), int(height * self.scale)
            img = self.image.resize((scaled_width, scaled_height), Image.Resampling.LANCZOS)
            self.tk_img = ImageTk.PhotoImage(img)
            self.canvas.delete("all")
            self.canvas.create_image(self.img_pos[0], self.img_pos[1], anchor=tk.NW, image=self.tk_img)
            self.canvas.config(scrollregion=self.canvas.bbox(tk.ALL))
            self.draw_points()

    def draw_points(self):
        self.canvas.delete("points")
        for index, target_widget in enumerate(self.app.target_widgets):
            coordinates = target_widget.get_coordinates()
            if coordinates:
                x, y = coordinates
                scaled_x, scaled_y = int(x * self.scale + self.img_pos[0]), int(y * self.scale + self.img_pos[1])
                color = "red" if self.app.radio_variable.get() == index else "blue"
                self.canvas.create_oval(scaled_x - 3, scaled_y - 3, scaled_x + 3, scaled_y + 3, fill=color, outline=color, tags="points")

    def on_double_click(self, event):
        if self.active_target is not None:
            canvas_x, canvas_y = self.canvas.canvasx(event.x), self.canvas.canvasy(event.y)
            img_x, img_y = int((canvas_x - self.img_pos[0]) / self.scale), int((canvas_y - self.img_pos[1]) / self.scale)
            self.app.update_coordinates(self.active_target, img_x, img_y)
            self.draw_points()

    def on_drag_start(self, event):
        self.canvas_origin = (event.x, event.y)

    def on_drag(self, event):
        dx = event.x - self.canvas_origin[0]
        dy = event.y - self.canvas_origin[1]
        self.canvas.move(tk.ALL, dx, dy)
        self.img_pos = (self.img_pos[0] + dx, self.img_pos[1] + dy)
        self.canvas_origin = (event.x, event.y)

    def on_zoom(self, event):
        scale_factor = 1.1 if event.delta > 0 else 0.9
        mouse_x = self.canvas.canvasx(event.x)
        mouse_y = self.canvas.canvasy(event.y)
        offset_x = (mouse_x - self.img_pos[0]) * (scale_factor - 1)
        offset_y = (mouse_y - self.img_pos[1]) * (scale_factor - 1)
        self.scale *= scale_factor
        self.img_pos = (self.img_pos[0] - offset_x, self.img_pos[1] - offset_y)
        self.update_image()


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Image Viewer")
        self.geometry("1200x800")  # Set initial window size
        self.minsize(600, 400)  # Set minimum window size

        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=0)
        self.rowconfigure(0, weight=1)

        main_frame = ttk.Frame(self)
        main_frame.grid(row=0, column=0, sticky="nsew")

        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(0, weight=1)

        # Image Frame
        image_frame = ttk.Frame(main_frame)
        image_frame.grid(row=0, column=0, sticky="nsew")
        self.image_viewer = ImageViewer(master=image_frame, app=self)

        # Control Panel
        control_panel = ttk.Frame(self)
        control_panel.grid(row=0, column=1, sticky="ns")
        self.control_panel = control_panel

        open_btn = ttk.Button(control_panel, text="Open Image/Video", command=self.image_viewer.open_image)
        open_btn.pack(pady=5)

        add_target_btn = ttk.Button(control_panel, text="Add Target", command=self.add_target)
        add_target_btn.pack(pady=5)

        save_btn = ttk.Button(control_panel, text="Save Target Coordinates", command=self.save_targets)
        save_btn.pack(pady=5)

        load_btn = ttk.Button(control_panel, text="Load Target Coordinates", command=self.load_targets)
        load_btn.pack(pady=5)

        self.radio_variable = tk.IntVar()
        self.radio_variable.set(-1)  # No target selected by default
        self.target_widgets = []

    def add_target(self):
        value = len(self.target_widgets)
        target_widget = TargetWidget(self.control_panel, variable=self.radio_variable, value=value, app=self)
        target_widget.pack(padx=5, pady=5)
        self.target_widgets.append(target_widget)

    def update_coordinates(self, target_index, x, y):
        if 0 <= target_index < len(self.target_widgets):
            self.target_widgets[target_index].update_coordinates(x, y)

    def set_active(self, target_index):
        if 0 <= target_index < len(self.target_widgets):
            self.image_viewer.active_target = target_index

    def save_targets(self):
        filepath = filedialog.asksaveasfilename(defaultextension=".csv")
        if filepath:
            with open(filepath, mode="w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(["Target", "X Pixels", "Y Pixels"])
                for target_widget in self.target_widgets:
                    writer.writerow(target_widget.get_target_data())

    def load_targets(self):
        filepath = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if not filepath:
            return

        with open(filepath, mode="r") as file:
            reader = csv.reader(file)
            next(reader)  # Skip the header

            # Remove existing target widgets
            for target_widget in self.target_widgets:
                target_widget.destroy()
            self.target_widgets.clear()

            # Create target widgets based on CSV and populate them
            for row in reader:
                target_name, x_pixels, y_pixels = row[0], row[1], row[2]
                self.add_target()
                self.target_widgets[-1].target_entry.insert(0, target_name)
                self.target_widgets[-1].xpixels_entry.insert(0, x_pixels)
                self.target_widgets[-1].ypixels_entry.insert(0, y_pixels)


if __name__ == "__main__":
    app = App()
    app.mainloop()
