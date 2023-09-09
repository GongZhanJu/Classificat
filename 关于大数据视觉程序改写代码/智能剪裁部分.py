import cv2
import numpy as np


def saliency_map(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Compute the saliency
    saliency = cv2.saliency.StaticSaliencySpectralResidual_create()
    _, saliency_map = saliency.computeSaliency(image)
    saliency_map = (saliency_map * 255).astype("uint8")

    return saliency_map


def smart_crop(image_path, output_dim=(100, 100)):
    sal_map = saliency_map(image_path)
    thresh_map = cv2.threshold(sal_map.astype('uint8'), 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    # Find the largest contour in the threshold map
    contours, _ = cv2.findContours(thresh_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    main_contour = max(contours, key=cv2.contourArea)

    # Get bounding box around the main contour
    x, y, w, h = cv2.boundingRect(main_contour)

    # Crop the original image
    original = cv2.imread(image_path)
    cropped = original[y:y + h, x:x + w]

    # Resize the cropped image to the desired output dimension
    final_output = cv2.resize(cropped, output_dim, interpolation=cv2.INTER_LINEAR)

    return final_output


# Example Usage:
input_image_path = "path_to_input_image.jpg"
cropped_image = smart_crop(input_image_path)
cv2.imwrite("path_to_save_cropped_image.jpg", cropped_image)
