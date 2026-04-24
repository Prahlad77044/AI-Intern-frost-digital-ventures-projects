import os
from torchvision import datasets
from PIL import Image

# -----------------------------
# Config
# -----------------------------
SELECTED_CLASSES = ["airplane", "automobile"]
NUM_IMAGES_PER_CLASS = 100
SAVE_FOLDER = "cifar_subset"
os.makedirs(SAVE_FOLDER, exist_ok=True)

# Load CIFAR-10 training dataset
cifar10_train = datasets.CIFAR10(root="./data", train=True, download=True)

# Map class names to indices
class_to_idx = {name: idx for idx, name in enumerate(cifar10_train.classes)}
selected_class_indices = [class_to_idx[c] for c in SELECTED_CLASSES]

# Counter for number of images saved per class
counter = {c: 0 for c in SELECTED_CLASSES}

# -----------------------------
# Save images of selected classes
# -----------------------------
for img, label in zip(cifar10_train.data, cifar10_train.targets):
    if label not in selected_class_indices:
        continue  # skip unwanted classes

    class_name = cifar10_train.classes[label]

    # Check if we already reached the limit
    if counter[class_name] >= NUM_IMAGES_PER_CLASS:
        continue

    # Save image
    class_folder = os.path.join(SAVE_FOLDER, class_name)
    os.makedirs(class_folder, exist_ok=True)
    img_pil = Image.fromarray(img)
    img_name = f"{class_name}_{counter[class_name]:03d}.png"
    img_pil.save(os.path.join(class_folder, img_name))

    counter[class_name] += 1

    # Stop early if all classes are done
    if all(counter[c] >= NUM_IMAGES_PER_CLASS for c in SELECTED_CLASSES):
        break

print("Saved subset of CIFAR-10 with two classes in:", SAVE_FOLDER)
