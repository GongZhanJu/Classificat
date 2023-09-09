import os
import tensorflow as tf
import requests
from sklearn.model_selection import train_test_split


# Your preprocessing functions are already defined.

# ... [rest of the preprocessing functions]

# 1. Resizing
def resize_image(image, height, width):
    return tf.image.resize(image, [height, width])

# 2. Normalization
def normalize_image(image):
    return image / 255.0

# 3. Histogram Equalization
def histogram_equalization(image):
    image = tf.image.rgb_to_yuv(image / 255.0)
    image_y, image_u, image_v = tf.split(image, 3, axis=-1)
    image_y = tf.image.per_image_standardization(image_y)
    return tf.image.yuv_to_rgb(tf.concat([image_y, image_u, image_v], axis=-1))

# 4. Data Augmentation (Random Horizontal Flip)
def random_flip(image):
    return tf.image.random_flip_left_right(image)

# 5. Gaussian Blurring
def gaussian_blur(image):
    blur_filter = tf.constant([
        [1, 2, 1],
        [2, 4, 2],
        [1, 2, 1]
    ], dtype=tf.float32) / 16.0
    blur_filter = tf.reshape(blur_filter, [3, 3, 1, 1])
    blur_channels = [blur_filter] * 3
    blur_kernel = tf.concat(blur_channels, axis=2)
    return tf.nn.depthwise_conv2d(image[tf.newaxis, ...], blur_kernel, strides=[1, 1, 1, 1], padding="SAME")[0]

def preprocess_image_from_path(image_path):
    # Read and decode the image from the path
    image = tf.io.read_file(image_path)
    image = tf.image.decode_png(image, channels=3)

    # Apply preprocessing
    image_resized = resize_image(image, 150, 150)
    image_normalized = normalize_image(image_resized)
    image_hist_eq = histogram_equalization(image_normalized)
    image_flipped = random_flip(image_hist_eq)
    image_blurred = gaussian_blur(image_flipped)

    # Save the preprocessed image back to the path
    tf.io.write_file(image_path, tf.image.encode_png(image_blurred))


# ... [rest of your code up to saving the images]

import os

# Create directories if they don't exist
category0_dir = os.path.join(image_dir, 'Category0')
category1_dir = os.path.join(image_dir, 'Category1')

if not os.path.exists(category0_dir):
    os.makedirs(category0_dir)

if not os.path.exists(category1_dir):
    os.makedirs(category1_dir)


# Function to download the image given its id (this function might need to be adjusted based on the backend API if the endpoint isn't provided)
def download_image(image_id):
    response = requests.get(f"{API_BASE_URL}image/download/{image_id}")
    return response.content


# Iterating over image records and saving images to respective directories
for record in image_records:
    image_content = download_image(record['id'])
    truth_id = record['truth_id']

    if id_code_mapping[truth_id] == '0':
        with open(os.path.join(category0_dir, f"{record['id']}.png"), 'wb') as f:
            f.write(image_content)
    else:
        with open(os.path.join(category1_dir, f"{record['id']}.png"), 'wb') as f:
            f.write(image_content)


from sklearn.model_selection import train_test_split

def split_data(directory):
    all_files = [os.path.join(directory, fname) for fname in os.listdir(directory)]
    train_files, test_files = train_test_split(all_files, test_size=0.2, random_state=42)
    train_files, val_files = train_test_split(train_files, test_size=0.25, random_state=42) # 0.25 x 0.8 = 0.2
    return train_files, val_files, test_files

# Preprocess images in Category0
for img_path in os.listdir(category0_dir):
    full_img_path = os.path.join(category0_dir, img_path)
    preprocess_image_from_path(full_img_path)

# Preprocess images in Category1
for img_path in os.listdir(category1_dir):
    full_img_path = os.path.join(category1_dir, img_path)
    preprocess_image_from_path(full_img_path)

# Now split your data into train, validation, and test sets
category0_train, category0_val, category0_test = split_data(category0_dir)
category1_train, category1_val, category1_test = split_data(category1_dir)
