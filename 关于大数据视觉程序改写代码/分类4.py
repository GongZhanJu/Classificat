import os
import requests
import tensorflow as tf
from sklearn.model_selection import train_test_split

# Set the image directory if it's not already set
image_dir = os.path.dirname(os.path.abspath(__file__))

# Create directories if they don't exist
category0_dir = os.path.join(image_dir, 'Category0')
category1_dir = os.path.join(image_dir, 'Category1')

if not os.path.exists(category0_dir):
    os.makedirs(category0_dir)

if not os.path.exists(category1_dir):
    os.makedirs(category1_dir)

# Function to download the image given its id
def download_image(image_id):
    response = requests.get(f"{API_BASE_URL}image/download/{image_id}")
    return response.content

# Iterating over image records and saving images to respective directories
for record in image_records:
    image_content = download_image(record['id'])
    truth_id = record['truth_id']

    if id_code_mapping[truth_id] == '0':
        file_path = os.path.join(category0_dir, f"{record['id']}.png")
    else:
        file_path = os.path.join(category1_dir, f"{record['id']}.png")

    with open(file_path, 'wb') as f:
        f.write(image_content)

    # Preprocess the saved image
    image = tf.io.read_file(file_path)
    image = tf.image.decode_png(image, channels=3)

    image = resize_image(image, 150, 150)
    image = normalize_image(image)
    image = histogram_equalization(image)
    image = random_flip(image)
    image = gaussian_blur(image)

    # Save the processed image
    tf.keras.preprocessing.image.save_img(file_path, image.numpy())

# Now split into train, val, and test sets
def split_data(directory):
    all_files = [os.path.join(directory, fname) for fname in os.listdir(directory)]
    train_files, test_files = train_test_split(all_files, test_size=0.2, random_state=42)
    train_files, val_files = train_test_split(train_files, test_size=0.25, random_state=42)  # 0.25 x 0.8 = 0.2
    return train_files, val_files, test_files

category0_train, category0_val, category0_test = split_data(category0_dir)
category1_train, category1_val, category1_test = split_data(category1_dir)
