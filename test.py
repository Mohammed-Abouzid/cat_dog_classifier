import cv2
from keras.models import load_model
import numpy as np
from keras.preprocessing import image

def test():

	CATEGORIES= ['Cat','Dog']

	classifier = load_model('cat_dog_classifier.h5')
	img_file = 'dog.jpg'        ##### change this	<<<<<<<<<<<<<<<<<<<<<
	test_image = image.load_img(img_file, target_size = (64, 64))     #resize img to 64*64
	test_image = image.img_to_array(test_image)
	test_image = np.expand_dims(test_image, axis = 0)
	result = classifier.predict(test_image)

	#print(CATEGORIES[int(result[0][0])])
	return CATEGORIES[int(result[0][0])]

if __name__ == "__main__":
	test()