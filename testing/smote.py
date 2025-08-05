#%%


#%%
train = pd.read_csv(config.CSV_DIR_TRAIN)
#to_smote = pd.DataFrame()

X = []
y = []

for row in tqdm(train):

	image_loader = get_data_loader(
			data_dir=config.DATADIR,
			dataset=row,
			mode=config.MODE,
			workers=8,
			batch_size=config.BATCH_SIZE,
			size_px=config.SIZE_PX,
			size_mm=config.SIZE_MM,
			rotations=config.ROTATION,
			translations=config.TRANSLATION,
			)

	#X = []
	#for data in tqdm(image_loader):
	data = next(iter(image_loader))  # Get the first batch
	label = data["label"][0].float().item()
	image = data["image"][0][0].numpy() #just one channel

	image_reshaped = image.reshape(1, -1)
	X.append(image_reshaped[0])
	y.append(label)

	#new_row = pd.DataFrame(image_reshaped, columns=[f'pixel_{i}' for i in range(image_reshaped.shape[1])])
	#new_row['label'] = label
	#to_smote = pd.concat([to_smote, new_row], ignore_index=True)

		# batch: shape (4930, 3, 64, 64)
		# labels = data["label"].cpu().numpy().reshape(-1, 1)
		# batch_first_channel = data["image"][:, 0, :, :].cpu().numpy()  # shape (4930, 64, 64)
		# batch_flattened = batch_first_channel.reshape(batch_first_channel.shape[0], -1)  # shape (4930, 4096)
		# X.append(np.hstack((batch_flattened, labels)))  # shape (4930, 4097)
#%%
smote = SMOTE(random_state=42)
X = to_smote.drop('label', axis=1)
y = to_smote['label']
X_resampled, y_resampled = smote.fit_resample(X, y)

resampled_df = pd.DataFrame(X_resampled, columns=X.columns)
resampled_df['label'] = y_resampled.values

#%%

#%%
resampled_df
# %%