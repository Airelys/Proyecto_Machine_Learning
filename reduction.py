import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
from vectorizer import take_data

warnings.filterwarnings("ignore")

def reduction ():

    data = take_data()
    dataset = []

    for user in data:
        for j in data[user]['vector_values']:
            X_features =j[:30]
            y_labels = j[31:]

            scaler = StandardScaler()
            X_features = scaler.fit_transform(X_features)

            pca = PCA()
            pca.fit_transform(X_features)
            pca_variance = pca.explained_variance_

            plt.figure(figsize=(8,6))
            plt.bar(range(len(j)), pca_variance, alpha=0.5, align='center', label="individual variance")
            plt.legend()
            plt.ylabel('Variance ratio')
            plt.xlabel('Principal components')
            plt.show()

            pca2 = PCA(n_components=10)
            pca2.fit(X_features)
            x_3d = pca.transform(X_features)

            plt.figure(figsize=(8,6))
            plt.scatter(x_3d[:,0], x_3d[:,5])
            plt.show()

            pca3 = PCA(n_components=2)
            pca3.fit(X_features)
            x_3d = pca3.transform(X_features)
            dataset.append(x_3d)

            plt.figure(figsize=(8,6))
            plt.scatter(x_3d[:,0], x_3d[:,1])
            plt.show()

    return dataset
