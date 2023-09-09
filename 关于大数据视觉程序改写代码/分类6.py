import os

# File to save paths of images with unknown formats
unknown_format_file = "unknown_format_images.txt"


def preprocess_image_from_path(image_path):
    # Read the image from the path
    image = tf.io.read_file(image_path)

    # Determine the decoding function based on file extension
    extension = os.path.splitext(image_path)[-1].lower()

    try:
        if extension == '.png':
            image_decoded = tf.image.decode_png(image, channels=3)
        elif extension in ['.jpg', '.jpeg']:
            image_decoded = tf.image.decode_jpeg(image, channels=3)
        elif extension == '.gif':
            image_decoded = tf.image.decode_gif(image)
        elif extension == '.bmp':
            image_decoded = tf.image.decode_bmp(image)
        else:
            raise ValueError(f"Unsupported file extension {extension} for image: {image_path}")
    except:
        with open(unknown_format_file, "a") as f:
            f.write(image_path + '\n')
        return

    # Continue with your preprocessing...
    image_resized = resize_image(image_decoded, 150, 150)
    image_normalized = normalize_image(image_resized)
    image_hist_eq = histogram_equalization(image_normalized)
    image_flipped = random_flip(image_hist_eq)
    image_blurred = gaussian_blur(image_flipped)

    # Save the preprocessed image back to the path
    tf.io.write_file(image_path, tf.image.encode_png(image_blurred))


# Make sure to clear the file before processing (in case the script is run multiple times)
if os.path.exists(unknown_format_file):
    os.remove(unknown_format_file)

# Rest of your code...
