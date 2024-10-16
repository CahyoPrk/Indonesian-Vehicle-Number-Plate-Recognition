import easyocr
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def detection(path, file_path_model):
    #import model
    model = tf.keras.models.load_model(file_path_model)
    #read
    image = load_img(path)
    image = np.array(image, dtype=np.uint8)
    image1 = load_img(path, target_size=(299,299))
    image_arr_299 = img_to_array(image1)/255.0
    h, w, d = image.shape
    test_arr = image_arr_299.reshape(1, 299, 299, 3)

    # prediksi
    coords = model.predict(test_arr)
    # denomalisasi
    denorm = np.array([w,w,h,h])
    coords = coords * denorm
    coords = coords.astype(np.int32)
    #draw Bounding Box
    xmin, xmax, ymin, ymax = coords[0]
    pt1 = (xmin, ymin)
    pt2 = (xmax, ymax)
    cv2.rectangle(image, pt1, pt2, (0, 255, 0), 3)
    return xmin, xmax, ymin, ymax, image

def crop_image(image, xmin, xmax, ymin, ymax):
    cropped_image = image[ymin:ymax, xmin:xmax]
    return cropped_image

def read_text_from_image(image):
    reader = easyocr.Reader(['en'])
    result = reader.readtext(image)
    detected_texts = [result[1] for result in result]
    return ' '.join(detected_texts[:3])