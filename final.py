import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, StackingRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import Ridge
import joblib

# Set seed for reproducibility
np.random.seed(123)

# Import training data

trn = pd.read_csv("cw1_train_eda.csv")
tst = pd.read_csv("cw1_test_eda.csv")

# Separate features (X) and target (y)
X_trn = trn.drop(columns=['outcome'])
y_trn = trn['outcome']
X_tst = tst

# Standardize features to have mean 0 and variance 1
scaler = StandardScaler()
X_trn = scaler.fit_transform(X_trn)
X_tst = scaler.transform(X_tst)

# Feature selection: Remove features with low variance
selector = VarianceThreshold(threshold=0.01)
X_trn = selector.fit_transform(X_trn)
X_tst = selector.transform(X_tst)

# Define hyperparameter search space for Gradient Boosting Regressor
parameters = {
    'n_estimators':np.arange(100,500,1000),
    'learning_rate': [0.001, 0.005, 0.01, 0.05, 0.1],
    'max_depth': [3,5,7,10],
    'min_samples_split': [2, 5, 10, 20],
    'min_samples_leaf': [1, 2, 4, 6, 10],
    'subsample': [0.7,0.8,0.9,1.0],
}

# Perform Randomized Search Cross-Validation to optimize Gradient Boosting Regressor
model = RandomizedSearchCV(
    estimator = GradientBoostingRegressor(random_state = 42), 
    param_distributions = parameters, 
    n_iter = 10, 
    cv=5,
    n_jobs=-1,
    verbose=1,
    random_state=42)  

# Train the model with the best hyperparameters
model.fit(X_trn, y_trn)
optimum_model = model.best_estimator_

# Define a Stacking Regressor combining Gradient Boosting and Ridge Regression
stacked_model = StackingRegressor(
    estimators=[('gbr', optimum_model), ('ridge', Ridge(alpha=1.0))],
    final_estimator=Ridge(alpha=1.0)
)

# Train the stacked model
stacked_model.fit(X_trn, y_trn)

# Generate predictions using the stacked model
yhat_gb = stacked_model.predict(X_tst)

# Format submission
out = pd.DataFrame({'yhat': yhat_gb})
out.to_csv('CW1_submission_k23051132.csv', index=False)

#Save Model
joblib.dump(stacked_model,'final_model.pkl')

