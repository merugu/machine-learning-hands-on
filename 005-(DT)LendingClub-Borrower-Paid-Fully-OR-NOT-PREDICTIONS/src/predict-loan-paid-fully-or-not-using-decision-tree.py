from lib2to3.pgen2.tokenize import Triple

import matplotlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix


def main():
    # Introduction to the program
    print("Using Decision Tree and Random Forest model from scikit-learn to predict if loan paid fully or not...")
    print("☕ Grab a cup of coffee and watch the magic happen!")

    # Load the dataset
    df = pd.read_csv('../data/loan_data.csv', index_col=0)
    pd.set_option('display.max_columns', None)
    # Print the first few rows of the dataframe
    print(df.head())

    cat_feats = ['purpose']

    final_data = pd.get_dummies(data=df, columns=cat_feats, drop_first=True)

    print(final_data)

    x = final_data.drop('not.fully.paid', axis=1)
    y = final_data['not.fully.paid']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=101)

    print('Using Decision Tree for predictions whether loan fully paid or not')
    dtree = DecisionTreeClassifier()
    dtree.fit(x_train, y_train)
    predictions_dt = dtree.predict(x_test)
    print('Printing Classification Report')
    print(classification_report(y_test, predictions_dt))
    dt_report = classification_report(y_test, predictions_dt, output_dict=True)

    print('Printing Confusion Metrics')
    print(confusion_matrix(y_test, predictions_dt))

    print('Using Random Forest Model for predictions; then we can compare which performed while whether decision tree or random forest')

    rf = RandomForestClassifier()
    rf.fit(x_train, y_train)
    prediction_rf = rf.predict(x_test)
    print('Printing Classification Report')
    print(classification_report(y_test, prediction_rf))
    rf_report  = classification_report(y_test, prediction_rf, output_dict=True)
    print('Printing Confusion Metrics')
    print(confusion_matrix(y_test, prediction_rf))

    dt_f1 = dt_report['weighted avg']['f1-score']
    rf_f1 = rf_report['weighted avg']['f1-score']

    # Print the results
    print("Decision Tree F1 Score:", dt_f1)
    print("Random Forest F1 Score:", rf_f1)

    # Determine the better model
    if dt_f1 > rf_f1:
        print("Decision Tree performs better.")
    elif rf_f1 > dt_f1:
        print("Random Forest performs better.")
    else:
        print("Both models perform equally well.")


if __name__ == "__main__":
    main()
