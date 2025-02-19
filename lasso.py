import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler

np.random.seed(123)

# Load dataset
df = pd.read_csv('cw1_train_eda.csv')


# Split dataset into training (80%) and testing (20%) sets
trn,tst = train_test_split(df,test_size=0.2,random_state=123)

# Separate features (X) and target (y)
X_trn = trn.drop(columns=['outcome'])
y_trn = trn['outcome']
X_tst = tst.drop(columns=['outcome'])
y_tst = tst['outcome']

# Standardize features using StandardScaler
scaler = StandardScaler()
X_trn = scaler.fit_transform(X_trn)
X_tst = scaler.transform(X_tst)

# Define hyperparameter grid for Lasso regression tuning
parameters = {
    'alpha': np.logspace(-4,2,50),
    'tol': [1e-4,1e-5],
    'max_iter': [1000,5000,10000],
    'selection':['cyclic','random']
}



# Perform grid search with 5-fold cross-validation to find the best hyperparameters
model = GridSearchCV(Lasso(),parameters,cv=5)
model.fit(X_trn, y_trn)

# Extract the best model from grid search
optimum_model = model.best_estimator_

# Make predictions on the test set
yhat_lasso = optimum_model.predict(X_tst)

# Format submission
out = pd.DataFrame({'yhat': yhat_lasso})
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