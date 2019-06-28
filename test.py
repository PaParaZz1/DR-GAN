import torch
import os
import numpy as np
import numpy
import sys
import time
from torchvision import transforms
from PIL import Image
sys.path.append('../options')
from test_options import TestOptions
sys.path.append('../data')
from data_loader import CreateDataLoader
sys.path.append('/../model')
from model_Loader import CreateModel
from Component import Tensor2Image
sys.path.append('../util')
from augmentation import *

if __name__ == "__main__":
	opt = TestOptions().parse()

	data_loader = CreateDataLoader(opt)
	model = CreateModel(opt)
	
	transform_std = transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))	
	])
	
	total = 0.	
	for i, data in enumerate(data_loader):
		real = Tensor2Image(data["image"][0])
		width, height = real.size
		result = Image.new(real.mode, (width*2, height*6))
		old = time.time()
		filename, syn = model.forward_np(data)
		run = time.time()-old
		total += run
		result.paste(real,(0,0,width,height))
		syn = Tensor2Image(syn[0])
		result.paste(syn, box=(width,0))
	    
		noise = add_gauss_noise(np.array(real))	
		data["image"][0] = transform_std(noise) 
		filename, syn = model.forward_np(data)
		result.paste(Image.fromarray(np.uint8(noise)),box=(0,height))
		syn = Tensor2Image(syn[0])
		result.paste(syn, box=(width,height))

		gamma = gamma_transform(np.array(real))	
		data["image"][0] = transform_std(gamma) 
		filename, syn = model.forward_np(data)
		result.paste(Image.fromarray(np.uint8(gamma)),box=(0,height*2))
		syn = Tensor2Image(syn[0])
		result.paste(syn, box=(width,height*2))
		
		shadow = shadow_transform(np.array(real))	
		data["image"][0] = transform_std(shadow) 
		filename, syn = model.forward_np(data)
		result.paste(Image.fromarray(np.uint8(shadow)),box=(0,height*3))
		syn = Tensor2Image(syn[0])
		result.paste(syn, box=(width,height*3))

		color = random_color(np.array(real))	
		data["image"][0] = transform_std(color) 
		filename, syn = model.forward_np(data)
		result.paste(Image.fromarray(np.uint8(color)),box=(0,height*4))
		syn = Tensor2Image(syn[0])
		result.paste(syn, box=(width,height*4))

		blur = blur_transform(np.array(real))	
		data["image"][0] = transform_std(blur) 
		filename, syn = model.forward_np(data)
		result.paste(Image.fromarray(np.uint8(blur)),box=(0,height*5))
		syn = Tensor2Image(syn[0])
		result.paste(syn, box=(width,height*5))

		filename = "{}".format(filename)
		path = os.path.join(opt.result_dir, filename)
		result.save(path)
		print(i)
	print(total)
	print(total/199)
