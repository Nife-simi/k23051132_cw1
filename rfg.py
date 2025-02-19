import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import Ridge

# Set seed for reproducibility
np.random.seed(123)

# Import training data
df = pd.read_csv('cw1_train_eda.csv')

# Random 80/20 train/validation split
trn, tst = train_test_split(df, test_size=0.2, random_state=123)

# Separate features (X) and target (y)
X_trn = trn.drop(columns=['outcome'])
y_trn = trn['outcome']
X_tst = tst.drop(columns=['outcome'])
y_tst = tst['outcome']

# Standardize features using StandardScaler
scaler = StandardScaler()
X_trn = scaler.fit_transform(X_trn)
X_tst = scaler.transform(X_tst)

# Feature selection
selector = VarianceThreshold(threshold=0.01)
X_trn = selector.fit_transform(X_trn)
X_tst = selector.transform(X_tst)

# Define hyperparameter search space for Random Forest
parameters = {
    'n_estimators': np.arange(100,1001,100),
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf':[1, 2, 4], 
    'max_features': ['sqrt','log2',None],
    'bootstrap': [True,False],
    'max_depth': [None, 10, 20],            
    'criterion': ['squared_error', 'absolute_error', 'poisson', 'friedman_mse']
}

# Train a Random Forest Regressor with hyperparameter tuning
model = RandomizedSearchCV(
    estimator = RandomForestRegressor(random_state=42),
    param_distributions= parameters,
    n_iter = 20,
    cv = 5,
    n_jobs = -1,
    verbose = 2,
    random_state = 42)
  
# Fit the model to the training data
model.fit(X_trn, y_trn)

# Retrieve the best model from hyperparameter tuning
optimum_model = model.best_estimator_

# Create a stacked model combining Random Forest and Ridge Regression
stacked_model = StackingRegressor(
    estimators=[('rf',optimum_model),('ridge', Ridge(alpha=0.1))],
    final_estimator = Ridge(alpha=1.0)

)

# Train the stacked model
stacked_model.fit(X_trn,y_trn)

# Make predictions using the stacked model
yhat_rf = stacked_model.predict(X_tst)

# Format submission
out = pd.DataFrame({'yhat': yhat_rf})
out.to_csv('CW1_submission.csv', index=False)

### THIS IS HOW THE SOLUTION IS EVALUATED ###

# Read in the submission
yhat_lm = np.array(pd.read_csv('CW1_submission.csv')['yhat'])

# This is the R^2 function
def r2_fn(yhat):
    eps = y_tst - yhat
    rss = np.sum(eps ** 2)
    tss = np.sum((y_tst - y_tst.mean()) ** 2)
    r2 = 1 - (rss / tss)
    return r2

# Evaluate
print(r2_fn(yhat_lm))