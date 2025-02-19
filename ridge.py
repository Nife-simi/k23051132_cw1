import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler

# Set seed
np.random.seed(123)

# Import training data
df = pd.read_csv('cw1_train_eda.csv')

# Random 80/20 train/validation split
trn, tst = train_test_split(df, test_size=0.2, random_state=123)

# Separate features (X) and target variable (y) for both training and test sets
X_trn = trn.drop(columns=['outcome'])
y_trn = trn['outcome']
X_tst = tst.drop(columns=['outcome'])
y_tst = tst['outcome']

# Standardize the feature values
scaler = StandardScaler()
X_trn = scaler.fit_transform(X_trn)
X_tst = scaler.transform(X_tst)

# Define hyperparameter grid for Ridge regression
parameters = {'alpha': np.logspace(-4, 2, 50),
               'solver': ['auto','saga'],
               'max_iter': [1000, 5000, 10000], 
    'tol': [1e-4, 1e-5]}                    

# Perform grid search cross-validation to find the best hyperparameters
model = GridSearchCV(Ridge(), parameters, scoring='r2', cv=5)

# Train the Ridge regression model using the best hyperparameters
model.fit(X_trn, y_trn)

# Retrieve the best Ridge model from the grid search
optimum_model = model.best_estimator_

# Make predictions using the best model on the test data
yhat_lm = optimum_model.predict(X_tst)

# Format submission
out = pd.DataFrame({'yhat': yhat_lm})
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




