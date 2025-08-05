from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch

from models.model_2d import ResNet18
from training.experiment_config import config
from training.dataloader import get_data_loader

from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


volumes_dir = Path(os.getenv("VOLUME_DIR"))
exp = "googlecpu_biggerinput_10e_64bs"

if __name__ == '__main__':
	# load model
	model = ResNet18()
	device = torch.device('cpu')
	model.load_state_dict(torch.load(
		f"results\{exp}\\best_metric_model.pth",
		map_location='cpu'
	))
	model.eval()

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

	outputs = []
	outputs_raw = []
	labels = []

	for data in tqdm(image_loader):

		label = data["label"].float().to(device).item()

		inputs = data["image"]
		input_tensor = inputs.to(device)

		model_output = model(input_tensor) #.detach().cpu().numpy()
		model_output = torch.sigmoid(model_output[0]).item()
		outputs_raw.append(model_output)
		model_output = round(model_output)

		outputs.append(model_output)
		labels.append(label)

	outputs = np.array(outputs).reshape(len(outputs), -1)
	labels = np.array(labels).reshape(-1, 1)
	
	confusion = confusion_matrix(labels, outputs)
	confusion_norm = confusion_matrix(labels, outputs, normalize='all')

	fpr, tpr, _ = roc_curve(labels, outputs_raw)
	auc_metric = auc(fpr, tpr)
	
	# Plots
	plt.figure(figsize=(10, 8))
	sns.heatmap(confusion, annot=True, fmt='d', cmap='Blues',
				xticklabels=['Predicted 0', 'Predicted 1'],
				yticklabels=['Actual 0', 'Actual 1'])
	plt.xlabel('Predicted Label')
	plt.ylabel('True Label')
	plt.title('Confusion Matrix\nAUC: {:.2f}'.format(auc_metric))
	plt.savefig(f'testing\confusion_matrices\{exp}_og_df.png')

	# plt.figure(figsize=(10, 8))
	# sns.heatmap(confusion_norm, annot=True, cmap='Blues', #fmt='d',
	# 			xticklabels=['Predicted 0', 'Predicted 1'],
	# 			yticklabels=['Actual 0', 'Actual 1'])
	# plt.xlabel('Predicted Label')
	# plt.ylabel('True Label')
	# plt.title('Confusion Matrix')
	# plt.savefig(f'testing\confusion_matrices\{exp}_norm.png')