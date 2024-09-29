import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import seaborn as sns
from sklearn import metrics


def main():
    # Introduction to the program
    print("🏠 House Price Prediction Using Linear Regression Model from scikit-learn...")
    print("☕ Grab a cup of coffee and watch the magic happen!")

    # Load the dataset
    df = pd.read_csv('../data/EcommerceCustomers.csv')
    pd.set_option('display.max_columns', None)
    print(df.head())
    x = df[['Avg. Session Length', 'Time on App','Time on Website', 'Length of Membership']]
    y = df['Yearly Amount Spent']
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=101)
    lrm = LinearRegression()
    lrm.fit(x_train, y_train)

    # Evaluate the model
    print("\n🔍 Model evaluation:")
    print(f"🟢 Intercept (constant term): {lrm.intercept_:.2f}")
    print("🟢 Coefficients for each feature:")

    # Create a DataFrame for coefficients
    coeff_df = pd.DataFrame(lrm.coef_, x.columns, columns=['Coefficient'])
    print(coeff_df)
    print("\n🔎 Interpretation of coefficients:")
    for feature, coef in zip(x.columns, lrm.coef_):
        print(f"🔹 A one unit increase in '{feature}' is associated with an increase of ${coef:.2f} in the Ecommerce price.")

    predictions = lrm.predict(x_test)
    print("\n🔎 Ecommerce price predictions:")
    print(predictions)
    # To know how far are we from the actual price compared to predictions using the  linear regression model;
    # we us can scatter plot from matplotlib with y_test which never used to train and predictions are the actual prediction using lrm
    plt.xlabel('Y Test (True Values)')
    plt.ylabel('Y label Predicted Value')
    print(plt.scatter(y_test, predictions))

    # Print the Histogram of residuals; I see it is normally distributed which is good sign that we choosed correct model for the data
    print(sns.displot((y_test - predictions)))

    # The three common Regression evaluation metrics are Mean Absolute Error(MAE), Mean Squared Error(MSE), Root Mean Squared error
    # The lower we minimize the error the best the model
    print('MAE')
    print(
        metrics.mean_absolute_error(y_test, predictions)
    )
    print('MSE')
    print(metrics.mean_squared_error(y_test, predictions))
    print('Root MSE')
    print(np.sqrt(metrics.mean_squared_error(y_test, predictions)))

    print("Explain the variance score, I'm getting around 98% ;; it is very good fit model on test data")
    print(metrics.explained_variance_score(y_test, predictions))

    print("Explore the residuals; Residuals means the difference between the values predicted by a model and values observed")
    sns.displot(y_test-predictions, bins=50)






if __name__ == "__main__":
    main()
