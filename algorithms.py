import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from reduction import reduction
from sklearn import svm
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix

import warnings

warnings.filterwarnings("ignore")

#def run_algorithms():

n_samples = 300
outliers_fraction = 0.15
n_outliers = int(outliers_fraction*n_samples)
n_inliers = n_samples - n_outliers

non_supervised_algorithms = [
    ("One Class SVM", svm.OneClassSVM(nu=outliers_fraction, kernel='rbf', gamma=0.1)),
    ("Robust covariance", EllipticEnvelope(contamination=outliers_fraction)),
    ("Isolation Forest", IsolationForest(contamination=outliers_fraction, random_state=42))
]

supervised_algorithms = [("GaussianNB", GaussianNB()),
                        ("KNeighborsClassifier", KNeighborsClassifier()),
                        ("DecisionTreeClassifier", DecisionTreeClassifier(criterion = "entropy")),
                        ("RandomForestClassifier", RandomForestClassifier(criterion = "entropy"))]

names, dataset = reduction()

print(names)



rng = np.random.RandomState(42)


plt.figure(figsize=(8,6))
plt.subplots_adjust(left=.02, right=.98, bottom=.001, top=.96, wspace=.05,
                    hspace=.01)
plot_num = 1
count = 0 


# Puesta en marcha de los algoritmos de aprendizaje no supervisado:


for X in dataset:
    for name,algorithm in non_supervised_algorithms:
        print(name + '\n')
        t0 = time.time()
        algorithm.fit(X)
        t1 = time.time()
        plt.subplot(len(dataset), len(non_supervised_algorithms), plot_num)
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


# Puesta en marcha de los algoritmos de aprendizaje supervisado

y_for_evaluated = []

for i in range(len(dataset)):
    y_for_evaluated.append([i] * len(dataset[i]))

X_for_supervised = []
y_for_supervised = []

for i in range(len(dataset)):
    for j in range(len(dataset[i])):
        X_for_supervised.append(dataset[i][j])
    y_for_supervised = y_for_supervised + y_for_evaluated[i]


X_train, X_test, y_train, y_test = train_test_split(X_for_supervised, y_for_supervised, train_size = 0.7)


for name, algorithm in supervised_algorithms:
    
    algorithm.fit(X_train, y_train)
    
    if len(names) < 6:
        disp = plot_confusion_matrix(algorithm, X_test, y_test,
                                display_labels=names,
                                cmap=plt.cm.Blues,
                                normalize='true')
        disp.ax_.set_title("Algorithm Used: " + name + "\n" + "General_Score: " + str(algorithm.score(X_test, y_test)))
        
        plt.show()
    else:
        print(name + " has score: " + str(algorithm.score(X_test, y_test)))

