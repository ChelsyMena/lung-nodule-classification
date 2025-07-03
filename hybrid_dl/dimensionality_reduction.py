import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn import random_projection
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import os
from dotenv import load_dotenv
load_dotenv()
working_dir = os.getenv("WORKING_DIR")

data = np.loadtxt(
	working_dir + "\\hybrid_dl" + "\\features_train_vgg.csv",
	delimiter=',', skiprows=1)

X = data[:, :-1]

transformer = random_projection.SparseRandomProjection(eps=0.3)
X = transformer.fit_transform(X)
print(X.shape)

# save new dataset
np.savetxt(
	working_dir + "\\hybrid_dl" + "\\new_features_train_vgg.csv",
	np.hstack([X, data[:, -1:]]),
	delimiter=',',
	header=','.join([f'feat_{i}' for i in range(X.shape[1])]) + ',label',
	comments=''
)

# Dim reduction for visualization
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X)

tsne = TSNE(n_components=3, random_state=42, perplexity=30)
X_tsne = tsne.fit_transform(X)

# save plots
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c=data[:, -1], cmap='viridis', marker='o')
plt.legend(['Benign', 'Malignant'], loc='upper right')
ax.set_xlabel('PCA 1')
ax.set_ylabel('PCA 2')
ax.set_zlabel('PCA 3')
plt.title('PCA of New Features')
plt.savefig("hybrid_dl/pca_features.png")

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_tsne[:, 0], X_tsne[:, 1], X_tsne[:, 2], c=data[:, -1], cmap='viridis', marker='o')
plt.legend(['Benign', 'Malignant'], loc='upper right')
ax.set_xlabel('t-SNE 1')
ax.set_ylabel('t-SNE 2')
ax.set_zlabel('t-SNE 3')
plt.title('t-SNE of New Features')
plt.savefig("hybrid_dl/tsne_features.png")