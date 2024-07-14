import os
import shutil
import random
from PIL import Image
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, array_to_img
import scipy  # Ensure SciPy is imported


### BUG Layback_spin and not_layback_spin folder names are reversed
def create_directories():
    paths = [
        'dataset/layback_spin',
        'dataset/not_layback_spin',
        'processed_dataset/train/layback_spin',
        'processed_dataset/train/not_layback_spin',
        'processed_dataset/val/layback_spin',
        'processed_dataset/val/not_layback_spin',
        'processed_dataset/test/layback_spin',
        'processed_dataset/test/not_layback_spin'
    ]
    for path in paths:
        os.makedirs(path, exist_ok=True)

def resize_images(input_dir, output_dir, size=(224, 224)):
    os.makedirs(output_dir, exist_ok=True)
    for filename in os.listdir(input_dir):
        img_path = os.path.join(input_dir, filename)
        if img_path.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
            img = Image.open(img_path)
            img = img.resize(size, Image.LANCZOS)  # Use Image.LANCZOS instead of Image.ANTIALIAS
            img.save(os.path.join(output_dir, filename))

def normalize_image(image):
    image_array = img_to_array(image)
    return image_array / 255.0

def augment_images(input_dir, output_dir, datagen, num_augmented_images=5):
    os.makedirs(output_dir, exist_ok=True)
    for filename in os.listdir(input_dir):
        img_path = os.path.join(input_dir, filename)
        if img_path.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
            img = Image.open(img_path)
            img = img.resize((224, 224), Image.LANCZOS)  # Use Image.LANCZOS instead of Image.ANTIALIAS
            x = img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = x / 255.0  # Normalize
            original_img_path = os.path.join(output_dir, filename)
            array_to_img(x[0]).save(original_img_path)
            i = 0
            for batch in datagen.flow(x, batch_size=1, save_to_dir=output_dir, save_prefix='aug', save_format='jpg'):
                i += 1
                if i >= num_augmented_images:
                    break

def split_dataset(src_dir, train_dir, val_dir, test_dir, split_ratio=(0.8, 0.1, 0.1)):
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    all_files = os.listdir(src_dir)
    if not all_files:
        print(f"No files found in {src_dir}")
    else:
        print(f"Found {len(all_files)} files in {src_dir}")
    random.shuffle(all_files)
    total_files = len(all_files)
    train_split = int(total_files * split_ratio[0])
    val_split = int(total_files * split_ratio[1]) + train_split
    train_files = all_files[:train_split]
    val_files = all_files[train_split:val_split]
    test_files = all_files[val_split:]
    def copy_files(files, dest_dir):
        for file in files:
            shutil.copy(os.path.join(src_dir, file), dest_dir)
            print(f"Copied {file} to {dest_dir}")
    copy_files(train_files, train_dir)
    copy_files(val_files, val_dir)
    copy_files(test_files, test_dir)

def preprocess_and_augment(input_dir, output_dir, size=(224, 224), num_augmented_images=5):
    os.makedirs(output_dir, exist_ok=True)
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    for filename in os.listdir(input_dir):
        img_path = os.path.join(input_dir, filename)
        if img_path.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
            img = Image.open(img_path)
            img = img.resize(size, Image.LANCZOS)  # Use Image.LANCZOS instead of Image.ANTIALIAS
            x = img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = x / 255.0  # Normalize
            original_img_path = os.path.join(output_dir, filename)
            array_to_img(x[0]).save(original_img_path)
            print(f"Saved original image {filename} to {output_dir}")
            i = 0
            for batch in datagen.flow(x, batch_size=1, save_to_dir=output_dir, save_prefix='aug', save_format='jpg'):
                i += 1
                if i >= num_augmented_images:
                    break

# Print the current working directory
print(f"Current working directory: {os.getcwd()}")

# List the contents of the dataset directory
print(f"Contents of 'dataset' directory: {os.listdir('dataset')}")

# Create necessary directories
create_directories()

# Split dataset into training, validation, and testing sets
split_dataset('dataset/layback_spin', 'processed_dataset/train/layback_spin', 'processed_dataset/val/layback_spin', 'processed_dataset/test/layback_spin')
split_dataset('dataset/not_layback_spin', 'processed_dataset/train/not_layback_spin', 'processed_dataset/val/not_layback_spin', 'processed_dataset/test/not_layback_spin')

# Preprocess and augment images
preprocess_and_augment('processed_dataset/train/layback_spin', 'processed_dataset/train/layback_spin')
preprocess_and_augment('processed_dataset/train/not_layback_spin', 'processed_dataset/train/not_layback_spin')
preprocess_and_augment('processed_dataset/val/layback_spin', 'processed_dataset/val/layback_spin')
preprocess_and_augment('processed_dataset/val/not_layback_spin', 'processed_dataset/val/not_layback_spin')
preprocess_and_augment('processed_dataset/test/layback_spin', 'processed_dataset/test/layback_spin')
preprocess_and_augment('processed_dataset/test/not_layback_spin', 'processed_dataset/test/not_layback_spin')



