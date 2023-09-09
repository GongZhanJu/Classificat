import tensorflow as tf


def preprocess_image(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_png(image, channels=3)  # Assuming PNG images

    # Resizing
    image = tf.image.resize(image, [150, 150])

    # Histogram Equalization
    image = tf.image.rgb_to_yuv(image / 255.0)  # Convert to float and YUV
    image_y, image_u, image_v = tf.split(image, 3, axis=-1)
    image_y = tf.image.per_image_standardization(image_y)
    image = tf.concat([image_y, image_u, image_v], axis=-1)
    image = tf.image.yuv_to_rgb(image)

    # Data Augmentation
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    image = tf.image.random_brightness(image, 0.2)

    # For rotation, we'll need to implement it ourselves or use external libraries if tf doesn't have it.
    # Skipping for now.
    # image = tf.image.random_rotation(image, 0.2)

    # Skipping zoom for now.
    # image = tf.image.random_zoom(image, (0.8, 1.2))

    # Gaussian Blur
    blur_filter = tf.constant([
        [1, 2, 1],
        [2, 4, 2],
        [1, 2, 1]
    ], dtype=tf.float32) / 16.0
    blur_filter = tf.reshape(blur_filter, [3, 3, 1, 1])
    blur_channels = [blur_filter] * 3
    blur_kernel = tf.concat(blur_channels, axis=2)
    image = tf.nn.depthwise_conv2d(image[tf.newaxis, ...], blur_kernel, strides=[1, 1, 1, 1], padding="SAME")[0]

    return image

