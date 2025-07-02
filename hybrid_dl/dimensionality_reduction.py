import numpy as np
from sklearn.decomposition import PCA, KernelPCA
from sklearn import random_projection

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import os
from dotenv import load_dotenv
load_dotenv()
working_dir = os.getenv("WORKING_DIR")

data = np.loadtxt(
	working_dir + "\hybrid_dl" + "\\features_with_labels_train.csv",
	delimiter=',', skiprows=1)

X = data[:, :-1]

#pca = KernelPCA(n_components=10, kernel='linear')
#X_transformed = pca.fit_transform(X)

transformer = random_projection.GaussianRandomProjection(eps=0.5)
X = transformer.fit_transform(X)
print(X.shape)

pca = PCA(n_components=3)
X_pca = pca.fit_transform(X)
#print(pca.explained_variance_ratio_)

# plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c=data[:, -1], cmap='viridis', marker='o')
plt.legend(['Malignant', 'Benign'], loc='upper right')
ax.set_xlabel('PCA 1')
ax.set_ylabel('PCA 2')
ax.set_zlabel('PCA 3')
plt.title('PCA of Features')
plt.savefig("hybrid_dl/pca_features.png")