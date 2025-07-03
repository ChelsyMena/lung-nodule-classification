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

from dotenv import load_dotenv
import os
from pathlib import Path
load_dotenv()
volumes_dir = Path(os.getenv("VOLUME_DIR"))
#working_dir = os.getenv("WORKING_DIR")


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
	
	desired_index = 27
	data = next(islice(image_loader, desired_index, None))
	label = data["label"].float().to(device)
	image_location = volumes_dir / "luna25_nodule_blocks" / "image" / f"{data['ID'][0]}.npy"

	inputs = data["image"]
	input_tensor = inputs.to(device)
	targets = [ClassifierOutputTarget(0)]

	volume = np.load(str(image_location))
	middle_idx = volume.shape[0] // 2
	middle_slice = volume[middle_idx, :, :]
	orig_img = np.stack([middle_slice]*3, axis=-1)
	orig_img = orig_img.astype(np.float32)
	orig_img = (orig_img - orig_img.min()) / (orig_img.max() - orig_img.min() + 1e-8)
	h, w = orig_img.shape[:2]
	 

	# input_img = inputs[0].detach().cpu().numpy()  
	# input_img = np.transpose(input_img, (1, 2, 0))
	# input_img = (input_img - input_img.min()) / (input_img.max() - input_img.min() + 1e-8)
	# input_img = input_img.astype(np.float32)

	# Construct the CAM object once, and then re-use it on many images.
	with GradCAM(model=model, target_layers=target_layers) as cam:
		# You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
		grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
		# In this example grayscale_cam has only one image in the batch:
		grayscale_cam = grayscale_cam[0, :]
		# input_img.shape: (H, W, C)
		h, w = orig_img.shape[:2]
		# Upsample grayscale_cam to input image size
		grayscale_cam_resized = cv2.resize(grayscale_cam, (w, h), interpolation=cv2.INTER_LINEAR)
		visualization = show_cam_on_image(orig_img, grayscale_cam_resized, use_rgb=True)
		# You can also get the model outputs without having to redo inference
		model_outputs = cam.outputs

		plt.imshow(visualization)
		#plt.axis('off')
		plt.title(f'Grad-CAM on nodule {data["ID"][0]}\nlabel: {label.item()}, model output: {model_outputs[0].item():.0f}')
		plt.imsave("testing\grad_cam_result.png", visualization, dpi=300)
		plt.show()

