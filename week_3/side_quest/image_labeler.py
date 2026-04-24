import os
import csv
import random
from tkinter import Tk, Label, Button, messagebox
from PIL import Image, ImageTk, UnidentifiedImageError
from tkinter import filedialog

# ----------------------------
# Configuration
# ----------------------------
LABELS_FILE = "labels.csv"
SPLITS_FILE = "splits.csv"
TRAIN_RATIO = 0.7
VAL_RATIO = 0.2
TEST_RATIO = 0.1
CLASS_NAMES = ["airplane", "automobile"]  # changed from Class 0/1

# ----------------------------
# Globals
# ----------------------------
image_paths = []
current_index = 0
labels = {}  # filename -> label
photo_label = None
btn_class0 = None
btn_class1 = None

# ----------------------------
# Functions
# ----------------------------
def open_folder():
    global image_paths, current_index, labels, btn_class0, btn_class1
    folder = filedialog.askdirectory(title="Select Image Folder")
    if not folder:
        return

    all_files = os.listdir(folder)
    print("Files in folder:", all_files)  # Debug

    # Filter images and ignore hidden files
    image_paths = [os.path.join(folder, f.strip()) for f in all_files
                   if not f.startswith(".") and f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".jfif"))]

    print("Filtered images:", image_paths)  # Debug

    if not image_paths:
        messagebox.showerror("No Images", "No images found in the selected folder.")
        return

    random.shuffle(image_paths)  # load images in random order
    current_index = 0

    # Load existing labels if present
    if os.path.exists(LABELS_FILE):
        with open(LABELS_FILE, "r") as f:
            reader = csv.DictReader(f)
            labels.update({row["filename"]: int(row["label"]) for row in reader})

    # Filter out already labeled images
    image_paths[:] = [p for p in image_paths if os.path.basename(p) not in labels]

    if not image_paths:
        messagebox.showinfo("Info", "All images in this folder are already labeled!")
        return

    # Show first image
    show_image()

    # Enable labeling buttons after folder is loaded
    btn_class0.pack(pady=5)
    btn_class1.pack(pady=5)

def show_image():
    global photo_label, current_index
    if current_index >= len(image_paths):
        messagebox.showinfo("Done", "Finished labeling all images in this folder!")
        return
    try:
        img = Image.open(image_paths[current_index]).convert("RGB")
        img = img.resize((400, 400))
        photo = ImageTk.PhotoImage(img)
        photo_label.configure(image=photo)
        photo_label.image = photo
        root.title(f"Image {current_index + 1}/{len(image_paths)}: {os.path.basename(image_paths[current_index])}")
    except UnidentifiedImageError:
        messagebox.showwarning("Image Error", f"Cannot open {image_paths[current_index]}")
        skip_image()

def save_label(label):
    global current_index, labels
    if current_index >= len(image_paths):
        return
    filename = os.path.basename(image_paths[current_index])
    labels[filename] = label
    save_labels_csv()
    current_index += 1
    if current_index < len(image_paths):
        show_image()
    else:
        messagebox.showinfo("Done", "Finished labeling all images in this folder!")

def skip_image():
    global current_index
    current_index += 1
    if current_index < len(image_paths):
        show_image()

def save_labels_csv():
    with open(LABELS_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "label"])
        for k, v in labels.items():
            writer.writerow([k, v])

def export_splits():
    if not labels:
        messagebox.showwarning("No labels", "Label some images before exporting splits.")
        return
    files = list(labels.keys())
    random.shuffle(files)

    n = len(files)
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)
    n_test = n - n_train - n_val

    splits = {"train": files[:n_train],
              "val": files[n_train:n_train + n_val],
              "test": files[n_train + n_val:]}

    with open(SPLITS_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "split"])
        for split_name, split_files in splits.items():
            for file in split_files:
                writer.writerow([file, split_name])
    messagebox.showinfo("Done", f"Splits saved to {SPLITS_FILE}")

# ----------------------------
# GUI Setup
# ----------------------------
root = Tk()
root.geometry("600x700")
root.title("Mini Image Labeler")

photo_label = Label(root)
photo_label.pack(pady=10)

btn_open = Button(root, text="Open Folder", command=open_folder, width=25, height=2)
btn_open.pack(pady=10)

btn_class0 = Button(root, text=f"{CLASS_NAMES[0]}", command=lambda: save_label(0),
                    width=25, height=2, bg="lightblue")
btn_class1 = Button(root, text=f"{CLASS_NAMES[1]}", command=lambda: save_label(1),
                    width=25, height=2, bg="lightgreen")

# Buttons will be packed after folder selection

btn_export = Button(root, text="Export Splits", command=export_splits, width=25, height=2)
btn_export.pack(pady=10)

# Keyboard shortcuts
def key_pressed(event):
    if event.char.lower() == "a":
        save_label(0)
    elif event.char.lower() == "b":
        save_label(1)

root.bind("<Key>", key_pressed)

root.mainloop()
