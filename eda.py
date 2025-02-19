import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore
from scipy.stats import boxcox
import warnings
warnings.filterwarnings('ignore')

def load_data(file):
    '''Loads the dataset from a CSV file'''
    df = pd.read_csv(file)
    print(df.head())
    print(df.info())
    return df

def encode(df,categorical_cols):
    '''One-hot encodes categorical variables'''
    return pd.get_dummies(df, columns=categorical_cols, drop_first=True)

def check_duplicates_and_nulls(df):
    """Checks for duplicates and null values in the dataset."""
    print("Unique values per column:")
    print(df.nunique())
    print("Null values per column:")
    print(df.isnull().sum())

def create_volume_feature(df):
    '''Creates a new volume feature and removes original dimensions'''
    df['volume'] = df['x'] * df['y'] * df['z']
    return df.drop(columns=['x', 'y', 'z'])

def display_correlation(df,target):
    '''Displays correlation matrix with respect to the target variables'''
    corr_matrix = df.corr()
    print(corr_matrix['outcome'].sort_values(ascending=False))

def transform_features(df,cols_to_transform):
    '''Applies log(1 + x) transformations to specified features.'''
    for col in cols_to_transform:
        df[col]=np.log1p(df[col])
    return df

def check_skewness(df,cols_to_check):
    '''Prints skewness of selected features with box plots'''
    for col in cols_to_check:
        print(f"{col} Skew: {round(df[col].skew(), 2)}")
        plt.figure(figsize=(15, 4))
        plt.subplot(1, 2, 1)
        df[col].hist(grid=False)
        plt.ylabel('count')
        plt.subplot(1, 2, 2)
        sns.boxplot(x=df[col])
        plt.show()

def remove_outliers(df):
    """Removes rows where any feature has a Z-score > 3."""
    z_scores = np.abs(zscore(df.select_dtypes(include=[np.number])))
    return df[(z_scores < 3).all(axis=1)]

def save_data(df, file_path):
    """Saves processed dataset to a CSV file."""
    df.to_csv(file_path, index=False)

def main():
    #file_path = "CW1_train.csv"
    #df = load_data(file_path)
    df_test = load_data('CW1_test.csv')
    #df_test = pd.read_csv('CW1_test.csv')
    
    categorical_cols = ['cut', 'color', 'clarity']
    #df = encode(df, categorical_cols)
    df_test = encode(df_test,categorical_cols)
    
    #check_duplicates_and_nulls(df)
    
    #df = create_volume_feature(df)
    df_test = create_volume_feature(df_test)

    #display_correlation(df, target='outcome')
    
    cols_to_transform = ["carat", "price", "volume"]
    #df = transform_features(df, cols_to_transform)
    df_test = transform_features(df_test,cols_to_transform)
    
    #num_cols = df.select_dtypes(include=np.number).columns.tolist()
    #check_skewness(df, num_cols)
    
    #df = remove_outliers(df)
    #check_skewness(df, num_cols)
    
    #save_data(df, 'cw1_train_eda.csv')
    save_data(df_test, 'cw1_test_eda.csv')
    print("Preprocessing complete. Data saved to 'cw1_train_eda.csv'.")

    train_df = pd.read_csv('cw1_train_eda.csv')
    print("Train shape:", train_df.shape)
    print("Test shape:", df_test.shape)
    print("Train first rows:\n", train_df.head())
    print("Test first rows:\n", df_test.head())

if __name__ == "__main__":
    main()

