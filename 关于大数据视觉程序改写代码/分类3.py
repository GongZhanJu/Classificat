import tensorflow as tf

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

# Example usage
image_path = './example_image.png'  # Replace with your image path
image = tf.io.read_file(image_path)
image = tf.image.decode_png(image, channels=3)

# Apply preprocessing
image_resized = resize_image(image, 150, 150)
image_normalized = normalize_image(image_resized)
image_hist_eq = histogram_equalization(image_normalized)
image_flipped = random_flip(image_hist_eq)
image_blurred = gaussian_blur(image_flipped)
