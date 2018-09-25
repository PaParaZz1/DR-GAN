import numpy as np
import numpy
from PIL import Image,ImageEnhance
import cv2
import random
def add_gauss_noise(img_path, mean = 0, std = 16):
	if isinstance(img_path, str):
		image = Image.open(img_path)
		img = np.array(image)
	elif isinstance(img_path, numpy.ndarray):
		img = img_path
	else:
		raise TypeError("Invalid input type")
	r = img[:,:,0].flatten()
	g = img[:,:,1].flatten()
	b = img[:,:,2].flatten()
	for i in range(img.shape[0]*img.shape[1]):
		r[i] += random.gauss(mean, std)
		g[i] += random.gauss(mean, std)
		b[i] += random.gauss(mean, std)
	img[:,:,0] = r.reshape(img.shape[0], img.shape[1])
	img[:,:,1] = g.reshape(img.shape[0], img.shape[1])
	img[:,:,2] = b.reshape(img.shape[0], img.shape[1])
	img.clip(0, 255)
	if isinstance(img_path, str):
		return Image.fromarray(np.uint8(img))
	else:
		return np.uint8(img)

def gamma_transform(img_path, gamma_param = 0.9):
	if isinstance(img_path, str):
		image = Image.open(img_path)
		img = np.array(image)
	elif isinstance(img_path, numpy.ndarray):
		img = img_path
	else:
		raise TypeError("Invalid input type")
	img = np.power(img, gamma_param)
	img.clip(0, 255)
	if isinstance(img_path, str):
		return Image.fromarray(np.uint8(img))
	else:
		return np.uint8(img)

def random_color(img_path):
	if isinstance(img_path, str):
		image = Image.open(img_path)
	elif isinstance(img_path, numpy.ndarray):
		image = Image.fromarray(np.uint8(img_path))
	else:
		raise TypeError("Invalid input type")
	factor = random.randint(0, 31)/10
	saturation = ImageEnhance.Color(image).enhance(factor)
	factor = random.randint(10, 21)/10
	brighten = ImageEnhance.Brightness(saturation).enhance(factor)
	factor = random.randint(10, 21)/10
	contrast = ImageEnhance.Contrast(brighten).enhance(factor)
	factor = random.randint(0, 31)/10
	sharpen = ImageEnhance.Sharpness(contrast).enhance(factor)
	if isinstance(img_path, str):
		return sharpen
	else:
		return np.array(sharpen)

def filp_horizontal(img_path):
	image = Image.open(img_path)
	return image.transpose(Image.FLIP_LEFT_RIGHT)

def shadow_transform(img_path, radius = 20):
	if isinstance(img_path, str):
		image = Image.open(img_path)
		img = np.array(image)
	elif isinstance(img_path, numpy.ndarray):
		img = img_path
	else:
		raise TypeError("Invalid input type")
	x = random.randint(radius, img.shape[0]-radius)
	y = random.randint(radius, img.shape[1]-radius)
	for i in range(x-radius//2, x+radius//2):
		for j in range(y-radius//2, y+radius//2):
			img[i,j,0]=0
			img[i,j,1]=0
			img[i,j,2]=0
	if isinstance(img_path, str):
		return Image.fromarray(np.uint8(img))
	else:
		return np.uint8(img)

def blur_transform(img_path, kernel = (5,5), std = 2):
	if isinstance(img_path, str):
		img = cv2.imread(img_path)
	elif isinstance(img_path, numpy.ndarray):
		img_path = img_path.astype(np.uint8)
		img = cv2.cvtColor(img_path, cv2.COLOR_RGB2BGR)
	else:
		raise TypeError("Invalid input type")
	img = cv2.GaussianBlur(img, kernel, std)
	if isinstance(img_path, str):
		return img
	else:
		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		return np.asarray(img)

if __name__ == "__main__":
	path = "test.png"
	im = add_gauss_noise(path)
	im.save("noise.png")
	im = gamma_transform(path)
	im.save("gamma.png")
	im = filp_horizontal(path)
	im.save("filp.png")
	im = blur_transform(path)
	cv2.imwrite("blur.png", im)
	im = shadow_transform(path)
	im.save("shadow.png")
	im = random_color(path)
	im.save("color.png")
