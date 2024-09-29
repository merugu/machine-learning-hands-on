import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression


def main():
    # Introduction to the program
    print("Titanic Survived or Not Prediction Using Logistic Regression Model from scikit-learn...")
    print("☕ Grab a cup of coffee and watch the magic happen!")

    # Load the dataset
    df = pd.read_csv('../data/titanic_train.csv')
    pd.set_option('display.max_columns', None)

    #Clean up the data we don't need below columns to predict is survivals
    df = df.drop(['PassengerId', 'Name','Ticket', 'Embarked', 'Cabin'], axis=1)
    df['male'] = pd.get_dummies(df['Sex'], drop_first=True)
    df = df.drop(['Sex'], axis=1)
    print(df.head())
    # remove all the row contains Nan values
    df = df.dropna()
    x = df.drop('Survived', axis=1)
    y= df['Survived']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=101)
    logrm = LogisticRegression()
    logrm.fit(x_train, y_train)
    predictions = logrm.predict(x_test)
    cf = confusion_matrix(y_test, predictions)
    cm_df = pd.DataFrame(cf,
                         index=['Actual Negative', 'Actual Positive'],
                         columns=['Predicted Negative', 'Predicted Positive'])
    print(cm_df.head())

if __name__ == "__main__":
    main()
