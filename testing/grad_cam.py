import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
from models.model_2d import ResNet18
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
from training.experiment_config import config
from training.dataloader import get_data_loader
from itertools import islice
import cv2

if __name__ == "__main__":

	# load best model
	model = ResNet18()
	device = torch.device('cpu')
	model.load_state_dict(torch.load(
		r"results\baseline2D_adam10epochs\best_metric_model.pth",
		map_location='cpu'
	))

	target_layers = [model.resnet18.layer4[-1]]

	valid = pd.read_csv(config.CSV_DIR_VALID)
	image_loader = get_data_loader(
		data_dir=config.DATADIR,
		dataset=valid,
		mode=config.MODE,
		workers=8,
		batch_size=config.BATCH_SIZE,
		size_px=config.SIZE_PX,
		size_mm=config.SIZE_MM,
		rotations=config.ROTATION,
		translations=config.TRANSLATION,
		)

	
	desired_index = 61
	data = next(islice(image_loader, desired_index, None))
	label = data["label"].float().to(device)
	inputs = data["image"]
	input_tensor = inputs.to(device)

	#input_tensor = # Create an input tensor image for your model..
	# Note: input_tensor can be a batch tensor with several images!

	# We have to specify the target we want to generate the CAM for.
	targets = [ClassifierOutputTarget(0)]

	# Assuming inputs is a batch tensor: (B, C, H, W)
	input_img = inputs[0].detach().cpu().numpy()  # Take the first image in the batch
	input_img = np.transpose(input_img, (1, 2, 0))  # (C, H, W) -> (H, W, C)
	input_img = (input_img - input_img.min()) / (input_img.max() - input_img.min() + 1e-8)  # Normalize to [0, 1]
	input_img = input_img.astype(np.float32)

	# Construct the CAM object once, and then re-use it on many images.
	with GradCAM(model=model, target_layers=target_layers) as cam:
		# You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
		grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
		# In this example grayscale_cam has only one image in the batch:
		grayscale_cam = grayscale_cam[0, :]
		# input_img.shape: (H, W, C)
		h, w = input_img.shape[:2]
		# Upsample grayscale_cam to input image size
		grayscale_cam_resized = cv2.resize(grayscale_cam, (w, h), interpolation=cv2.INTER_LINEAR)
		visualization = show_cam_on_image(input_img, grayscale_cam_resized, use_rgb=True)
		# You can also get the model outputs without having to redo inference
		model_outputs = cam.outputs

		plt.imshow(visualization)
		plt.axis('off')
		plt.title('Grad-CAM')
		plt.show()
		plt.imsave("testing\grad_cam_result.png", visualization, dpi=300)

