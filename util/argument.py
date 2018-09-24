import numpy as np
from PIL import Image,ImageEnhance
import cv2
import random
def add_gauss_noise(img_path, mean = 0, std = 16):
	image = Image.open(img_path)
	img = np.array(image)
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
	return Image.fromarray(np.uint8(img))

def gamma_transform(img_path, gamma_param = 0.9):
	image = Image.open(img_path)
	img = np.array(image)
	img = np.power(img, gamma_param)
	img.clip(0, 255)
	return Image.fromarray(np.uint8(img))

def random_color(img_path):
	image = Image.open(img_path)
	factor = random.randint(0, 31)/10
	saturation = ImageEnhance.Color(image).enhance(factor)
	factor = random.randint(10, 21)/10
	brighten = ImageEnhance.Brightness(saturation).enhance(factor)
	factor = random.randint(10, 21)/10
	contrast = ImageEnhance.Contrast(brighten).enhance(factor)
	factor = random.randint(0, 31)/10
	sharpen = ImageEnhance.Sharpness(contrast).enhance(factor)
	return sharpen 

def filp_horizontal(img_path):
	image = Image.open(img_path)
	return image.transpose(Image.FLIP_LEFT_RIGHT)

def shadow(img_path, radius = 20):
	image = Image.open(img_path)
	img = np.array(image)
	x = random.randint(radius, img.shape[0]-radius)
	y = random.randint(radius, img.shape[1]-radius)
	for i in range(x-radius//2, x+radius//2):
		for j in range(y-radius//2, y+radius//2):
			img[i,j,0]=0
			img[i,j,1]=0
			img[i,j,2]=0
	return Image.fromarray(np.uint8(img))

def blur(img_path, kernel = (5,5), std = 2):
	img = cv2.imread(img_path)
	img = cv2.GaussianBlur(img, kernel, std)
	return img

if __name__ == "__main__":
	path = "test.png"
	im = add_gauss_noise(path)
	im.save("noise.png")
	im = gamma_transform(path)
	im.save("gamma.png")
	im = filp_horizontal(path)
	im.save("filp.png")
	im = blur(path)
	cv2.imwrite("blur.png", im)
	im = shadow(path)
	im.save("shadow.png")
	im = random_color(path)
	im.save("color.png")
