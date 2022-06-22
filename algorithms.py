import time
import numpy as np

from vectorizer import take_data, vectorizer
from sklearn import svm
from sklearn.datasets import make_moons,make_blobs
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

n_samples = 300
outliers_fraction = 0.15
n_outliers = int(outliers_fraction*n_samples)
n_inliers = n_samples - n_outliers

algorithms = [
    ("One Class SVM", svm.OneClassSVM(nu=outliers_fraction, kernel='rbf', gamma=0.1)),
    ("Robust covriance", EllipticEnvelope(contamination=outliers_fraction)),
    ("Isolation Forest", IsolationForest(contamination=outliers_fraction, random_state=42))
    #("Local Outlier Factor", LocalOutlierFactor(n_neighbors=35, contamination=outliers_fraction))
]

dataset = take_data()
lists = {'user': {'text': ["ale feo","te odio","no te soporto mas"]}}
y_tets = vectorizer(lists)['user']['vector_values'][0]

xx, yy = np.meshgrid(np.linspace(-7,7,150),
                     np.linspace(-7,7,150)) 

rng = np.random.RandomState(42)

for user in dataset:
    X = dataset[user]['vector_values'][0]
    #X = np.concatenate([dataset[user]['vector_values'], rng.uniform(low=-6, high=6, size=(n_outliers,2))], axis=0)

    print(len(X[0]))
    for name,algorithm in algorithms:
        print(name + '/n')
        t0 = time.time()
        algorithm.fit(X)
        t1 = time.time()

        #if name == "Local Outlier Factor":
        y_predict = algorithm.fit_predict(y_tets)
        print("y_predict /n" + str(y_predict))
        #else:
            #y_predict = algorithm.fit(X).predict(y_tets)
            #print("y_predict " + str(y_predict))

        '''if name != "Local Outlier Factor":
            Z = algorithm.predict(y_tets)
            #Z = Z.reshape(xx.shape)
            print("z /n" + str(Z))'''
