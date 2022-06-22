import time
import numpy as np
import matplotlib.pyplot as plt
from vectorizer import take_data, vectorizer
from reduction import reduction
from sklearn import svm
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
import warnings

warnings.filterwarnings("ignore")

n_samples = 300
outliers_fraction = 0.15
n_outliers = int(outliers_fraction*n_samples)
n_inliers = n_samples - n_outliers

algorithms = [
    ("One Class SVM", svm.OneClassSVM(nu=outliers_fraction, kernel='rbf', gamma=0.1)),
    ("Robust covriance", EllipticEnvelope(contamination=outliers_fraction)),
    ("Isolation Forest", IsolationForest(contamination=outliers_fraction, random_state=42))
]


dataset = reduction()


rng = np.random.RandomState(42)

xx = np.meshgrid(np.linspace(-7, 7, 2))
yy = np.meshgrid(np.linspace(-7, 7, 2))

plt.figure(figsize=(8,6))
plt.subplots_adjust(left=.02, right=.98, bottom=.001, top=.96, wspace=.05,
                    hspace=.01)
plot_num = 1
count = 0 

for i in dataset:
    X = i
    for name,algorithm in algorithms:
        print(name + '\n')
        t0 = time.time()
        algorithm.fit(X)
        t1 = time.time()
        plt.subplot(len(dataset), len(algorithms), plot_num)
        if count == 0:
            plt.title(name, size=18)

        y_predict = algorithm.fit(X).predict(X)
        print("y_predict \n" + str(y_predict))


        colors = np.array(['#377eb8', '#ff7f00'])
        plt.scatter(X[:, 0], X[:, 1], s=10, color=colors[(y_predict+1)//2 ])

        plt.xlim(-2, 2)
        plt.ylim(-2, 2)
        plt.xticks(())
        plt.yticks(())
        plt.text(.99, .01, ('%.2fs' % (t1 - t0)).lstrip('0'),
                 transform=plt.gca().transAxes, size=15,
                 horizontalalignment='right')
        plot_num += 1

    count += 1
plt.show()
