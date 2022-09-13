import os
from typing import Union

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt
from PIL import Image


try:
	get_ipython().run_line_magic('tensorflow_version', '2.x')

except Exception:
	pass

def get_eps_value() -> Union[float, None]:
	while True:
		try:
			eps: float = float(input("floating point values "))
			if (eps < 1 and eps > 0):
				return eps

			print("no more than 1 no less than 0")

		except KeyboardInterrupt:
			return None

		except:
			print("VALUE ONLY FLOAT")


def preprocess(image):
	image = tf.cast(image, tf.float32)
	image = image/255
	image = tf.image.resize(image, (224, 224) )
	image = image[None, ...]
	
	return image


def create_adversarial_pattern(input_image, input_label):
	with tf.GradientTape() as tape:
		tape.watch(input_image)
		prediction = pretrained_model(input_image)
		loss = loss_object(input_label, prediction)

	# Get the gradients of the loss w.r.t to the input image.
	gradient = tape.gradient(loss, input_image)
	# Get the sign of the gradients to create the perturbation
	signed_grad = tf.sign(gradient)
	
	return signed_grad


def image_save(image: list, file_name: str) -> None:
	#image = tf.image.resize(image[0], (300, 300), antialias=True)
	#image = tf.image.resize_with_pad(image[0], 483, 858)
	tf.keras.utils.save_img(f"result/{file_name}", image[0],)



eps = get_eps_value()

if eps is not None:
	files = os.listdir('example')
	pretrained_model = tf.keras.applications.MobileNetV2(include_top=True, weights='imagenet')
	pretrained_model.trainable = False
	decode_predictions = tf.keras.applications.mobilenet_v2.decode_predictions


	for file_name in files:
		img_format = file_name.split('.')[-1]

		if not img_format in ['jpg', 'jpeg', 'png']:
			print(f"{file_name} is not supported, supported format [jpg, jpeg, png]")
			continue

		image_raw = tf.io.read_file(f"example/{file_name}")	
		image = tf.image.decode_image(image_raw)

		image = preprocess(image)
		image_probs = pretrained_model.predict(image)
		loss_object = tf.keras.losses.CategoricalCrossentropy()

		perturbations = create_adversarial_pattern(image, image_probs)

		adv_x = image + eps*perturbations
		adv_x = tf.clip_by_value(adv_x, 0, 1)
		image_save(adv_x, file_name)
