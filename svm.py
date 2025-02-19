import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler

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

# Standardize the feature values
scaler = StandardScaler()
X_trn = scaler.fit_transform(X_trn)
X_tst = scaler.transform(X_tst)

# Define hyperparameter grid for Support Vector Regression (SVR)
parameters = {
    'C': [0.1,1,10,100],
    'epsilon': [0.01,0.1,0.5,1],
    'kernel': ['linear','poly','rbf'],
    'gamma': ['scale','auto'],
}

# Perform grid search cross-validation to find the best hyperparameters
model = GridSearchCV(SVR(),parameters,cv=5,n_jobs=-1,verbose=2)

# Train the SVR model using the best hyperparameters
model.fit(X_trn, y_trn)

# Retrieve the best SVR model from the grid search
optimum_model = model.best_estimator_

# Make predictions using the best model on the test data
yhat_svm = optimum_model.predict(X_tst)

# Format submission
out = pd.DataFrame({'yhat': yhat_svm})
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