import os
import requests
from PIL import Image, ImageOps, ImageEnhance
import numpy as np
from sklearn.model_selection import train_test_split


# ... (Your existing code for directory creation and function definitions)

def preprocess_and_save_images(image_list, save_dir):
    for image_path in image_list:
        img = Image.open(image_path)

        # Apply your preprocessing steps
        img_vertical_flipped = vertical_flip(img)
        img_horizontal_flipped = horizontal_flip(img)
        img_clipped = clip_image(img, 200, 200)
        img_normalized = normalize_image(img)

        # Save the preprocessed images
        base_name = os.path.basename(image_path).split('.')[0]
        img_vertical_flipped.save(os.path.join(save_dir, f"{base_name}_vertical_flipped.png"))
        img_horizontal_flipped.save(os.path.join(save_dir, f"{base_name}_horizontal_flipped.png"))
        img_clipped.save(os.path.join(save_dir, f"{base_name}_clipped.png"))

        # If you want to save the normalized image, you'll need to convert it back to a PIL Image
        img_normalized_pil = Image.fromarray((img_normalized * 255).astype('uint8'))
        img_normalized_pil.save(os.path.join(save_dir, f"{base_name}_normalized.png"))


# Get all image paths from each category
all_category0_images = [os.path.join(category0_dir, fname) for fname in os.listdir(category0_dir)]
all_category1_images = [os.path.join(category1_dir, fname) for fname in os.listdir(category1_dir)]

# Preprocess and save images for each category
preprocess_and_save_images(all_category0_images, category0_dir)
preprocess_and_save_images(all_category1_images, category1_dir)

# Split the data into training, validation, and test sets
category0_train, category0_val, category0_test = split_data(category0_dir)
category1_train, category1_val, category1_test = split_data(category1_dir)

# Now, category0_train, category0_val, category0_test, category1_train, category1_val, and category1_test
# contain the file paths for the training, validation, and test sets for each category.
