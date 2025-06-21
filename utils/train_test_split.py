from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

from dotenv import load_dotenv
import os

load_dotenv()

volumes_dir = os.getenv("VOLUME_DIR")
working_dir = os.getenv("WORKING_DIR")

if __name__ == "__main__":
	
	file_path = volumes_dir + "\LUNA25_Public_Training_Development_Data.csv"

	data = pd.read_csv(file_path, sep=',')
	#print(data.head())

	y = data.label
	X = data.drop(columns=['label'])

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

	# Save the split data
	train_data = pd.concat([X_train, y_train], axis=1)
	test_data = pd.concat([X_test, y_test], axis=1)

	train_data.to_csv(volumes_dir + "\\train.csv", index=False)
	test_data.to_csv(volumes_dir + "\\valid.csv", index=False)
