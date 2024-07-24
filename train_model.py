import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor
import joblib

# Load dataset
df = pd.read_csv('/mnt/data/Housing.csv')

# Check the column names
print(df.columns)

# Identify the columns to drop (if any)
# Assuming similar columns as before; adjust according to the new dataset
columns_to_drop = ['Alley', 'PoolQC', 'Fence', 'MiscFeature']
df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])

# Fill missing values with the median value for numerical columns and the mode for categorical columns
for column in df.columns:
    if df[column].dtype == 'object':
        df[column] = df[column].fillna(df[column].mode()[0])
    else:
        df[column] = df[column].fillna(df[column].median())

# Encode categorical variables
label_encoders = {}
for column in df.select_dtypes(include='object').columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

# Split the data into features and target
# Assuming 'SalePrice' is the target variable; adjust if different
X = df.drop(columns='SalePrice')
y = df['SalePrice']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the XGBoost model
xgb_model = XGBRegressor(n_estimators=1000, learning_rate=0.05, max_depth=6, random_state=42)
xgb_model.fit(X_train, y_train)

# Save the model and label encoders
joblib.dump(xgb_model, 'xgboost_model.pkl')
joblib.dump(label_encoders, 'label_encoders.pkl')
