import cv2
import os

def process_image(file_path):
    img = cv2.imread(file_path)
    grayscale_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    processed_path = file_path.replace('.jpg', '_processed.jpg')
    cv2.imwrite(processed_path, grayscale_img)
    return processed_path
