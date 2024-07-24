
import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
import numpy as np
from xgboost import XGBRegressor

# Load the data
file_path = r'C:\Users\veena\OneDrive\Desktop\Mini Project exec 1\Housing.csv'
data = pd.read_csv(file_path)

# Display the column names
st.write("Columns in the dataset:")
st.write(data.columns)

# Define a function to train and evaluate different regression models
def train_and_evaluate(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    return mse

# Streamlit application
def main():
    st.title("Housing Price Prediction")
    
    uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
    
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.write(data.head())

        if st.checkbox("Show Summary Statistics"):
            st.write(data.describe())

        # Display the column names
        st.write("Columns in the dataset:")
        st.write(data.columns)

        # Choose the target column
        target = st.selectbox("Select the target column", data.columns)
        
        # Choose regression model
        model_name = st.selectbox("Choose a regression model", list(models.keys()))

        if st.button("Train and Evaluate"):
            X = data.drop(columns=target)
            y = data[target]
            
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            model = models[model_name]
            mse = train_and_evaluate(model, X_train, X_test, y_train, y_test)
            st.write(f"Mean Squared Error for {model_name}: {mse}")
            
            # Plot the results
            predictions = model.predict(X_test)
            plt.figure(figsize=(10, 6))
            sns.scatterplot(x=y_test, y=predictions)
            plt.xlabel('True Values')
            plt.ylabel('Predictions')
            plt.title(f"{model_name} Predictions vs True Values")
            st.pyplot(plt)

if __name__ == "__main__":
    main()
