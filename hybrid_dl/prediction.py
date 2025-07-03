import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

# Load features and labels
df = pd.read_csv("hybrid_dl/new_features_train_vgg.csv")
X = df.drop(columns=["label"]).values
y = df["label"].values

# Split into train/validation for quick test
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train SVM (with probability=True for AUC)
# clf = SVC(kernel='sigmoid', probability=True, random_state=42)
# clf.fit(X_train, y_train)

# # Predict probabilities for validation set
# y_proba = clf.predict_proba(X_val)[:, 1]

# # Compute AUC
# auc = roc_auc_score(y_val, y_proba)
# print(f"SVM validation AUC: {auc:.3f}")

from sklearn.neighbors import KNeighborsClassifier

# Train KNN
best_auc = 0
best_k = None

for k in range(1, 100):  # Try k from 1 to 50
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_proba_knn = knn.predict_proba(X_val)[:, 1]
    auc_knn = roc_auc_score(y_val, y_proba_knn)
    print(f"K={k}: KNN validation AUC: {auc_knn:.3f}")
    if auc_knn > best_auc:
        best_auc = auc_knn
        best_k = k

print(f"\nBest K: {best_k} with validation AUC: {best_auc:.3f}")