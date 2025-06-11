import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import cv2
import subtractors as sub
import showAlpha as alph


class BackgroundSubtractionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Background Subtraction App")

        self.upload_button = tk.Button(self.root, text="Upload Foreground Image", command=self.upload_image)
        self.upload_button.pack()

        self.upload_bg_button = tk.Button(self.root, text="Upload Background Image", command=self.upload_background)
        self.upload_bg_button.pack()

        self.image_frame = tk.Frame(self.root)
        self.image_frame.pack()

        self.original_label = tk.Label(self.image_frame)
        self.original_label.pack(side=tk.LEFT, padx=10)

        self.processed_label = tk.Label(self.image_frame)
        self.processed_label.pack(side=tk.LEFT, padx=10)

        self.method = tk.StringVar(value="Automatic")
        self.method_label = tk.Label(self.root, text="Choose background subtraction method:")
        self.method_label.pack()

        self.method_options = ["Automatic", "Square Select", "Brush", "Real-time video", "Replace Background"]
        self.method_dropdown = ttk.Combobox(self.root, textvariable=self.method, values=self.method_options)
        self.method_dropdown.pack()

        self.process_button = tk.Button(self.root, text="Apply", command=self.apply_background_subtraction)
        self.process_button.pack()

        self.download_button = tk.Button(self.root, text="Download Image", command=self.download_image)
        self.download_button.pack()

        self.original_image = None
        self.background_image = None
        self.processed_image = None
        self.path = None
        self.bg_path = None

    def upload_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")])
        if file_path:
            self.original_image = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
            self.path = file_path
            self.show_side_by_side(self.original_image, None)

    def upload_background(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg")])
        if file_path:
            self.background_image = cv2.imread(file_path)
            self.bg_path = file_path
            messagebox.showinfo("Background Loaded", "Background image uploaded successfully.")

    def apply_background_subtraction(self):
        if self.method.get() == "Real-time video":
            self.processed_image = sub.camera_subtraction(self.path)
            return

        if self.original_image is None:
            messagebox.showerror("Error", "No image uploaded")
            return

        method = self.method.get()

        if method == "Automatic":
            self.processed_image = sub.automatic_subtraction(self.path)
        elif method == "Square Select":
            self.processed_image = sub.square_select_subtraction(self.path)
        elif method == "Brush":
            self.processed_image = sub.brush_subtraction(self.path)
        elif method == "Replace Background":
            if self.background_image is None:
                messagebox.showerror("Error", "No background image uploaded")
                return
            self.processed_image = sub.replace_background(self.original_image, self.background_image)
        else:
            messagebox.showerror("Error", "Invalid method selected")
            return

        
        if method in ["Automatic", "Square Select", "Brush", "Replace Background"]:
            if self.processed_image.shape[2] == 4:
                display_image = alph.show_image_with_alpha_cv2(self.processed_image)
            else:
                display_image = self.processed_image
        else:
            display_image = self.processed_image

        self.show_side_by_side(self.original_image, display_image)

    def download_image(self):
        if self.processed_image is None:
            messagebox.showerror("Error", "No processed image to save")
            return

        file_path = filedialog.asksaveasfilename(defaultextension=".png",
                                                 filetypes=[("PNG Files", "*.png"),
                                                            ("JPEG Files", "*.jpg"),
                                                            ("All Files", "*.*")])
        if file_path:
            cv2.imwrite(file_path, self.processed_image)
            messagebox.showinfo("Success", f"Image saved successfully to {file_path}")

    def show_side_by_side(self, original, processed):
        if original is not None:
            if original.shape[2] == 4:
                display = alph.show_image_with_alpha_cv2(original)
                original_rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
            else:
                original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)

            original_pil = Image.fromarray(original_rgb).resize((400, 400))
            original_img = ImageTk.PhotoImage(original_pil)
            self.original_label.config(image=original_img)
            self.original_label.image = original_img
        else:
            self.original_label.config(image=None)
            self.original_label.image = None

        if processed is not None:
            display_rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
            processed_pil = Image.fromarray(display_rgb).resize((400, 400))
            processed_img = ImageTk.PhotoImage(processed_pil)
            self.processed_label.config(image=processed_img)
            self.processed_label.image = processed_img


if __name__ == "__main__":
    root = tk.Tk()
    app = BackgroundSubtractionApp(root)
    root.mainloop()