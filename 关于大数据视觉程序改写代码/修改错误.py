import os
import requests
from PIL import Image, ImageOps, ImageEnhance
import numpy as np
from sklearn.model_selection import train_test_split


API_BASE_URL = "YOUR_API_BASE_URL"
image_dir = "./images"


category0_dir = os.path.join(image_dir, 'Category0')
category1_dir = os.path.join(image_dir, 'Category1')

if not os.path.exists(category0_dir):
    os.makedirs(category0_dir)

if not os.path.exists(category1_dir):
    os.makedirs(category1_dir)


def download_image(image_id):
    response = requests.get(f"{API_BASE_URL}image/download/{image_id}")
    return response.content


def split_data(directory):
    all_files = [os.path.join(directory, fname) for fname in os.listdir(directory)]
    train_files, test_files = train_test_split(all_files, test_size=0.2, random_state=42)
    train_files, val_files = train_test_split(train_files, test_size=0.25, random_state=42)  # 0.25 x 0.8 = 0.2
    return train_files, val_files, test_files


def vertical_flip(img: Image.Image) -> Image.Image:
    return ImageOps.flip(img)


def horizontal_flip(img: Image.Image) -> Image.Image:
    return ImageOps.mirror(img)


def clip_image(img: Image.Image, target_height: int, target_width: int) -> Image.Image:
    width, height = img.size
    left = (width - target_width) / 2
    top = (height - target_height) / 2
    right = (width + target_width) / 2
    bottom = (height + target_height) / 2
    return img.crop((left, top, right, bottom))


def normalize_image(img: Image.Image) -> np.ndarray:
    img_array = np.array(img)
    return img_array / 255.0



for record in image_records:
    image_content = download_image(record['id'])
    truth_id = record['truth_id']

    if id_code_mapping[truth_id] == '0':
        file_path = os.path.join(category0_dir, f"{record['id']}.png")
    else:
        file_path = os.path.join(category1_dir, f"{record['id']}.png")

    with open(file_path, 'wb') as f:
        f.write(image_content)

category0_train, category0_val, category0_test = split_data(category0_dir)
category1_train, category1_val, category1_test = split_data(category1_dir)


img = Image.open("path_to_image.jpg")
img_vertical_flipped = vertical_flip(img)
img_horizontal_flipped = horizontal_flip(img)
img_clipped = clip_image(img, 200, 200)
img_normalized = normalize_image(img)

img_vertical_flipped.save("path_to_save_vertical_flipped.jpg")
img_horizontal_flipped.save("path_to_save_horizontal_flipped.jpg")
img_clipped.save("path_to_save_clipped.jpg")
