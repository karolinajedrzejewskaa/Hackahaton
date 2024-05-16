import pandas as pd
from sklearn.model_selection import train_test_split
from category_encoders import TargetEncoder
from sklearn.preprocessing import OrdinalEncoder
import xgboost as xgb
import pickle
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error



# Load data
path = "/Users/lua/wild/Hackathon/tableau_numerique.csv"
df = pd.read_csv(path, sep=',')

# Define features and target
X = df[['work_setting', 'experience_level', 'work_year', 'company_size', 'job_title']]
y = df['salary_in_usd_log']  # Assuming the target variable is log-transformed

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and fit the encoders
target_encoder = TargetEncoder(cols=['job_title'])
target_encoder.fit(X_train['job_title'], y_train)

# Transform the job_title feature
X_train['job_title_encoded'] = target_encoder.transform(X_train['job_title'], y_train)
X_test['job_title_encoded'] = target_encoder.transform(X_test['job_title'])
X_train.drop('job_title', axis=1, inplace=True)
X_test.drop('job_title', axis=1, inplace=True)

# Initialize and fit the other encoders
work_setting_encoder = OrdinalEncoder(categories=[['Remote', 'Hybrid', 'In-person']])
experience_level_encoder = OrdinalEncoder(categories=[['Entry-level', 'Mid-level', 'Senior', 'Executive']])
year_encoder = OrdinalEncoder(categories=[[2020, 2021, 2022, 2023]])
company_size_encoder = OrdinalEncoder(categories=[['S', 'M', 'L']])

work_setting_encoder.fit(df[['work_setting']])
experience_level_encoder.fit(df[['experience_level']])
year_encoder.fit(df[['work_year']])
company_size_encoder.fit(df[['company_size']])

# Transform the data
X_train['work_setting'] = work_setting_encoder.transform(X_train[['work_setting']])
X_train['experience_level'] = experience_level_encoder.transform(X_train[['experience_level']])
X_train['work_year'] = year_encoder.transform(X_train[['work_year']])
X_train['company_size'] = company_size_encoder.transform(X_train[['company_size']])
X_test['work_setting'] = work_setting_encoder.transform(X_test[['work_setting']])
X_test['experience_level'] = experience_level_encoder.transform(X_test[['experience_level']])
X_test['work_year'] = year_encoder.transform(X_test[['work_year']])
X_test['company_size'] = company_size_encoder.transform(X_test[['company_size']])

# Train the XGBoost Regressor
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', colsample_bytree=0.3, learning_rate=0.1, max_depth=5, alpha=10, n_estimators=100)
xgb_model.fit(X_train, y_train)

# Save the trained model
with open('/Users/lua/wild/Hackathon/model.pkl', 'wb') as file:
    pickle.dump(xgb_model, file)

# Save the fitted encoders into a dictionary
encoders = {
    'work_setting': work_setting_encoder,
    'experience_level': experience_level_encoder,
    'work_year': year_encoder,
    'company_size': company_size_encoder,
    'job_title': target_encoder
}

with open('/Users/lua/wild/Hackathon/encoders.pkl', 'wb') as file:
    pickle.dump(encoders, file)

y_pred_xgb = xgb_model.predict(X_test)
print(r2_score(y_test, y_pred_xgb))
rmse_xgb = mean_squared_error(y_test, y_pred_xgb, squared=False)
print(f"XGBoost RMSE: {rmse_xgb}")