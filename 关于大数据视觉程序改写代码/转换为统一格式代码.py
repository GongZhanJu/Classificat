def preprocess_and_save_images(image_list, save_dir):
    for image_path in image_list:
        try:
            img = Image.open(image_path)

            # Apply your preprocessing steps
            img_vertical_flipped = vertical_flip(img)
            img_horizontal_flipped = horizontal_flip(img)
            img_clipped = clip_image(img, 200, 200)
            img_normalized = normalize_image(img)

            # Save the preprocessed images in JPG format
            base_name = os.path.basename(image_path).split('.')[0]
            img_vertical_flipped.save(os.path.join(save_dir, f"{base_name}_vertical_flipped.jpg"), "JPEG")
            img_horizontal_flipped.save(os.path.join(save_dir, f"{base_name}_horizontal_flipped.jpg"), "JPEG")
            img_clipped.save(os.path.join(save_dir, f"{base_name}_clipped.jpg"), "JPEG")

            # If you want to save the normalized image, you'll need to convert it back to a PIL Image
            img_normalized_pil = Image.fromarray((img_normalized * 255).astype('uint8'))
            img_normalized_pil.save(os.path.join(save_dir, f"{base_name}_normalized.jpg"), "JPEG")

        except Exception as e:
            print(f"Error processing {image_path}: {e}")
