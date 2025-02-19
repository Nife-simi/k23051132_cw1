import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, StackingRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import Ridge



# Set seed for reproducibility
np.random.seed(123)

# Load dataset
df = pd.read_csv('cw1_train_eda.csv')

# Random 80/20 train/validation split
trn, tst = train_test_split(df, test_size=0.2, random_state=123)


# Separate features (X) and target (y)
X_trn = trn.drop(columns=['outcome'])
y_trn = trn['outcome']
X_tst = tst.drop(columns=['outcome'])
y_tst = tst['outcome']

# Standardize features
scaler = StandardScaler()
X_trn = scaler.fit_transform(X_trn)
X_tst = scaler.transform(X_tst)

# Feature selection: Remove low variance features
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

# Perform Randomized Search CV to optimize Gradient Boosting Regressor
model = RandomizedSearchCV(
    estimator = GradientBoostingRegressor(random_state = 42), 
    param_distributions = parameters, 
    n_iter = 10, 
    cv=5,
    n_jobs=-1,
    verbose=1,
    random_state=42)  

# Train the model
model.fit(X_trn, y_trn)
optimum_model = model.best_estimator_

# Define and train a Stacking Regressor
stacked_model = StackingRegressor(
    estimators=[('gbr', optimum_model), ('ridge', Ridge(alpha=1.0))],
    final_estimator=Ridge(alpha=1.0)
)
stacked_model.fit(X_trn, y_trn)

# Generate predictions
yhat_gb = stacked_model.predict(X_tst)

# Format submission
out = pd.DataFrame({'yhat': yhat_gb})
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

#Evaluate
print(r2_fn(yhat_lm))
