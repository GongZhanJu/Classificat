import os
import tensorflow as tf
import requests
from sklearn.model_selection import train_test_split

# Preprocessing Functions (Merged as discussed)
def preprocess_image(image):
    image_resized = resize_image(image, 150, 150)
    image_normalized = normalize_image(image_resized)
    image_hist_eq = histogram_equalization(image_normalized)
    image_flipped = random_flip(image_hist_eq)
    image_blurred = gaussian_blur(image_flipped)
    return image_blurred

def preprocess_image_from_path(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_png(image, channels=3)
    processed_image = preprocess_image(image)
    tf.io.write_file(image_path, tf.image.encode_png(processed_image))

# Directory Creation
base_dir = os.path.join(image_dir, 'base')  # Raw images without preprocessing.
preprocessed_dir = os.path.join(image_dir, 'preprocessed')  # Preprocessed images.
for directory in [base_dir, preprocessed_dir]:
    for category in ['Category0', 'Category1']:
        full_path = os.path.join(directory, category)
        if not os.path.exists(full_path):
            os.makedirs(full_path)

# Downloading and Saving Images to base directory
for record in image_records:
    image_content = download_image(record['id'])
    truth_id = record['truth_id']
    category = 'Category0' if id_code_mapping[truth_id] == '0' else 'Category1'
    with open(os.path.join(base_dir, category, f"{record['id']}.png"), 'wb') as f:
        f.write(image_content)

# Preprocessing and Saving to Preprocessed Directory
for category in ['Category0', 'Category1']:
    raw_category_dir = os.path.join(base_dir, category)
    for img_path in os.listdir(raw_category_dir):
        full_img_path = os.path.join(raw_category_dir, img_path)
        preprocess_image_from_path(full_img_path)
        preprocessed_img_path = os.path.join(preprocessed_dir, category, img_path)
        tf.io.write_file(preprocessed_img_path, tf.image.encode_png(tf.image.decode_image(tf.io.read_file(full_img_path))))

# Splitting Data
def split_data(directory):
    all_files = [os.path.join(directory, fname) for fname in os.listdir(directory)]
    train_files, test_files = train_test_split(all_files, test_size=0.2, random_state=42)
    train_files, val_files = train_test_split(train_files, test_size=0.25, random_state=42)
    return train_files, val_files, test_files

category0_train, category0_val, category0_test = split_data(os.path.join(preprocessed_dir, 'Category0'))
category1_train, category1_val, category1_test = split_data(os.path.join(preprocessed_dir, 'Category1'))
