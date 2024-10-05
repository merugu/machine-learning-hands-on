from lib2to3.pgen2.tokenize import Triple

import matplotlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import classification_report, confusion_matrix


def main():
    # Introduction to the program
    print("Using Decision Tree and Random Forest model from scikit-learn to predict if loan paid fully or not...")
    print("☕ Grab a cup of coffee and watch the magic happen!")

    # Load the dataset
    cancer = load_breast_cancer()
    print(cancer['DESCR'])

    df_feats = pd.DataFrame(cancer['data'], columns=cancer['feature_names'])
    print(df_feats.head())

    print(cancer['target_names'])
    x = df_feats
    y = cancer['target']
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.33, random_state=101)

    svc = SVC()
    print('Printing fit parameters')
    svc.fit(x_train, y_train)
    predictions = svc.predict(x_test)

    print(confusion_matrix(y_test, predictions))
    print('\n')
    print(classification_report(y_test, predictions))

    #GridSearch allows you find the best parameters as possible ; it will use bunch of combination and get the best parameters as possible
    param_grid = {'C':[0.1,1,10,100,1000], 'gamma': [1,0.1,0.01,0.001,0.0001]}
    grid = GridSearchCV(SVC(), param_grid, verbose=3)
    grid.fit(x_train, y_train)
    print('Best Params')
    print('\n')
    print(grid.best_params_)

    print('Best Estimator')
    print('\n')
    print(grid.best_estimator_)

    grid_predictions = grid.predict(x_test)
    print('Confusion Matrix')
    print('\n')
    print(confusion_matrix(y_test, grid_predictions))
    print('\n')
    print('Classification Report')
    print('\n')
    print(classification_report(y_test, grid_predictions))











if __name__ == "__main__":
    main()
