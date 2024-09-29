import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


def main():
    # Introduction to the program
    print("🏠 House Price Prediction Using Linear Regression Model from scikit-learn...")
    print("☕ Grab a cup of coffee and watch the magic happen!")

    # Load the dataset
    df = pd.read_csv('../data/USA_Housing.csv')

    # Display the columns in the dataset
    print("\n📊 The following columns are available in the dataset:")
    print(df.columns.tolist())

    # Drop the 'Address' column as it is not needed for prediction
    df = df.drop('Address', axis=1)

    # Set display options to show all columns
    pd.set_option('display.max_columns', None)

    # Display a sample of the housing data
    print("\n🔍 Sample data from the housing dataset:")
    print(df.head())

    # Split the data into feature variables (X) and the target variable (y)
    print("\n📈 Splitting the data into features (X) and target (y)...")
    x = df[['Avg. Area Income', 'Avg. Area House Age',
            'Avg. Area Number of Rooms', 'Avg. Area Number of Bedrooms',
            'Area Population']]
    y = df['Price']

    print("✅ Features (X):", x.columns.tolist())
    print("✅ Target variable (y): 'Price'")

    # Split the data into training and testing sets
    print("\n🔄 Splitting the data into training and testing sets (33% for testing)...")
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=101)

    # Initialize and train the Linear Regression model
    print("\n📉 Training the Linear Regression model...")
    lrm = LinearRegression()
    lrm.fit(x_train, y_train)

    # Evaluate the model
    print("\n🔍 Model evaluation:")
    print(f"🟢 Intercept (constant term): {lrm.intercept_:.2f}")
    print("🟢 Coefficients for each feature:")

    # Create a DataFrame for coefficients
    coeff_df = pd.DataFrame(lrm.coef_, x.columns, columns=['Coefficient'])

    # Display the coefficients with explanations
    print(coeff_df)
    print("\n🔎 Interpretation of coefficients:")
    for feature, coef in zip(x.columns, lrm.coef_):
        print(f"🔹 A one unit increase in '{feature}' is associated with an increase of ${coef:.2f} in the house price.")


if __name__ == "__main__":
    main()
