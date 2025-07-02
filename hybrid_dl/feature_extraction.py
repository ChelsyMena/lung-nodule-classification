import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)
sys.path.append(project_root)

import pandas as pd
import numpy as np
from training.experiment_config import config
from training.dataloader import get_data_loader
from tqdm import tqdm

import torch
from dl_models import ResNet18

if __name__ == "__main__":

	train = pd.read_csv(config.CSV_DIR_TRAIN)
	valid = pd.read_csv(config.CSV_DIR_VALID)

	for df, split_name in zip([train, valid], ["train", "valid"]):

		image_loader = get_data_loader(
			data_dir=config.DATADIR,
			dataset=df,
			mode=config.MODE,
			workers=8,
			batch_size=config.BATCH_SIZE,
			size_px=config.SIZE_PX,
			size_mm=config.SIZE_MM,
			rotations=config.ROTATION,
			translations=config.TRANSLATION,
		)

		model = ResNet18()
		device = torch.device('cpu')
		outputs = []
		labels = []

		for  i, data in enumerate(tqdm(image_loader, desc=f"Extracting {split_name}")):

			label = data["label"].float().to(device)
			inputs = data["image"]
			inputs = inputs.to(device)
	
			output = model(inputs) #.detach().cpu().numpy()
			#label = data["label"].detach().cpu().numpy()

			outputs.append(output.detach().numpy())
			labels.append(label.detach().numpy())
		
		# save to csv
		outputs = np.array(outputs).reshape(len(outputs), -1)
		labels = np.array(labels).reshape(-1, 1)

		# print(outputs)
		# print(labels)

		features_and_labels = np.hstack([outputs, labels])

		feature_cols = [f'feat_{i}' for i in range(outputs.shape[1])]
		df_out = pd.DataFrame(features_and_labels, columns=feature_cols + ['label'])

		df_out.to_csv(config.WORKDIR +"\hybrid_dl" + f'features_with_labels_{split_name}.csv', index=False)