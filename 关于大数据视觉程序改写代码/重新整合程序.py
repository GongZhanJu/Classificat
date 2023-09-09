# ... [省略了导入部分和API、URL等定义]

# Splitting data
def split_data(directory):
    all_files = [os.path.join(directory, fname) for fname in os.listdir(directory)]
    train_files, test_files = train_test_split(all_files, test_size=0.2, random_state=42)
    train_files, val_files = train_test_split(train_files, test_size=0.25, random_state=42)
    return train_files, val_files, test_files

# New preprocessing functions
def color_jitter(img: Image.Image, brightness=0.2, contrast=0.2, saturation=0.2) -> Image.Image:
    img = ImageEnhance.Brightness(img).enhance(1 + brightness * (2 * np.random.random() - 1))
    img = ImageEnhance.Contrast(img).enhance(1 + contrast * (2 * np.random.random() - 1))
    img = ImageEnhance.Color(img).enhance(1 + saturation * (2 * np.random.random() - 1))
    return img

def resize_image(img: Image.Image, target_size=(224, 224)) -> Image.Image:
    return img.resize(target_size, Image.ANTIALIAS)

# ... [省略了之前的其他函数定义]

# Preprocessing function
def preprocess_and_save_images(image_list, save_dir):
    for image_path in image_list:
        try:
            img = Image.open(image_path)
            img_vertical_flipped = vertical_flip(img)
            img_horizontal_flipped = horizontal_flip(img)
            img_normalized = normalize_image(img)
            img_resized = resize_image(img)
            img_jittered = color_jitter(img)

            base_name = os.path.basename(image_path).split('.')[0]
            img_vertical_flipped.save(os.path.join(save_dir, f"{base_name}_vertical_flipped.png"))
            img_horizontal_flipped.save(os.path.join(save_dir, f"{base_name}_horizontal_flipped.png"))
            img_resized.save(os.path.join(save_dir, f"{base_name}_resized.png"))
            img_jittered.save(os.path.join(save_dir, f"{base_name}_jittered.png"))

            img_normalized_pil = Image.fromarray((img_normalized * 255).astype('uint8'))
            img_normalized_pil.save(os.path.join(save_dir, f"{base_name}_normalized.png"))
        except Exception as e:
            print(f'Error processing image {image_path}. Error: {e}')

# ... [省略了图像下载部分]

# Removing extra save statements from downloading images
for record in image_records:
    image_content = download_image(record['id'])
    truth_id = record['truth_id']

    # Note: `id_code_mapping` is not defined in the provided code.
    # Make sure it's properly defined or replaced by appropriate logic.
    if id_code_mapping[truth_id] == '0':
        file_path = os.path.join(category0_dir, f"{record['id']}.png")
    else:
        file_path = os.path.join(category1_dir, f"{record['id']}.png")

    with open(file_path, 'wb') as f:
        f.write(image_content)

# ... [省略了其他代码]
