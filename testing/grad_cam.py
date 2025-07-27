import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.base_cam import BaseCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

from models.model_2d import ResNet18
from training.experiment_config import config
from training.dataloader import get_data_loader

import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
#from itertools import islice

import os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()
volumes_dir = Path(os.getenv("VOLUME_DIR"))

class BinaryClassifierOutputTarget:
		def __call__(self, model_output):
			if model_output.ndim == 2:
				return model_output[:, 0]
			elif model_output.ndim == 1:
				return model_output
			else:
				raise ValueError(f"Unexpected model_output shape: {model_output.shape}")

if __name__ == "__main__":

	# load model
	model = ResNet18()
	device = torch.device('cpu')
	model.load_state_dict(torch.load(
		r"results\baseline2D_adam10epochs\best_metric_model.pth",
		map_location='cpu'
	))
	model.eval()
	target_layers = [model.resnet18.layer4[-1]]

	valid = pd.read_csv(config.CSV_DIR_VALID)
	valid = valid[valid["label"] == 1]

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

	#Construct the CAM object once, and then re-use it on many images.
	with GradCAM(model=model, target_layers=target_layers) as cam:

		for data in tqdm(image_loader):

			# desired_index = 0
			# data = next(islice(image_loader, desired_index, None))
			label = data["label"].float().to(device)
			image_location = volumes_dir / "luna25_nodule_blocks" / "image" / f"{data['ID'][0]}.npy"

			inputs = data["image"]
			input_tensor = inputs.to(device)
			input_tensor.requires_grad_()

			targets = [BinaryClassifierOutputTarget()]

			# Imagen original
			volume = np.load(str(image_location))
			middle_idx = volume.shape[0] // 2
			middle_slice = volume[middle_idx, :, :]
			orig_img = np.stack([middle_slice]*3, axis=-1)
			orig_img = orig_img.astype(np.float32)
			orig_img = (orig_img - orig_img.min()) / (orig_img.max() - orig_img.min() + 1e-8)
			h, w = orig_img.shape[:2]

			grayscale_cam = cam(
				input_tensor=input_tensor,
				targets=targets,
				eigen_smooth=True,
				aug_smooth=True
				)
			grayscale_cam = grayscale_cam[0, :]
			h, w = orig_img.shape[:2]

			image_np = data["image"][0].detach().cpu().numpy()
			image_np = np.transpose(image_np, (1, 2, 0))
			visualization = show_cam_on_image(image_np, grayscale_cam, use_rgb=True, image_weight=0.7)

			# grayscale_cam_resized = cv2.resize(grayscale_cam, (w, h), interpolation=cv2.INTER_CUBIC)
			# visualization = show_cam_on_image(orig_img, grayscale_cam_resized, use_rgb=True, image_weight=0.7)

			model_outputs = cam.outputs
			pred = round(torch.sigmoid(model_outputs[0]).item())

			#print("Grayscale CAM stats:", grayscale_cam.min(), grayscale_cam.max(), grayscale_cam.mean())

			#plt.imshow(visualization)
			#plt.axis('off')
			#plt.title(f'Grad-CAM on nodule {data["ID"][0]}\nlabel: {label.item()}, model output: {pred}')
			#plt.imsave("testing\resnet_gradcams\grad_cam_result.png", visualization, dpi=300)
			#plt.show()

			plt.subplot(1,3,2)
			plt.imshow(image_np, cmap='gray')
			plt.title("Model input")

			plt.subplot(1,3,1)
			plt.imshow(orig_img, cmap='gray')
			plt.title("Original image")

			plt.subplot(1,3,3)
			plt.imshow(visualization)
			plt.title("Grad-CAM visualization")

			plt.suptitle(f'Grad-CAM on nodule {data["ID"][0]}\nlabel: {label.item()}, model output: {pred}')
			plt.tight_layout()
			#plt.show()
			plt.savefig(f"testing/resnet_gradcams/ones/{data['ID'][0]}.png", dpi=300)