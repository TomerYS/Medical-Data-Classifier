import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from datetime import datetime

# All preprocessing steps for train and test datasets should be added here
def preprocess_data(personal_info_train, measurements_train, personal_info_test, measurements_test):
    # Merge datasets on patient_id
    train = pd.merge(personal_info_train, measurements_train, on='patient_id')
    test = pd.merge(personal_info_test, measurements_test, on='patient_id')

    # Identify duplicates in training dataset
    duplicateRowsDF = train[train.duplicated(['patient_id'])]

    # Check if duplicates exist
    if duplicateRowsDF.empty:
        print('No duplicates found.')
    else:
        print("Duplicate Rows based on 'patient_id' are:", duplicateRowsDF, sep='\n')

    # Remove duplicates
    train.drop_duplicates(subset='patient_id', keep='first', inplace=True)

    # Convert gender to numeric values using a map
    gender_map = {'M': 0, 'F': 1}
    train['gender'] = train['gender'].map(gender_map)
    test['gender'] = test['gender'].map(gender_map)

    # height and weight columns have some values that are too high or too low to be realistic
    for df in [train, test]:
        df.loc[df['height'] < 10, 'height'] = df.loc[df['height'] < 10, 'height'] * 100

    for df in [train, test]:
        df.loc[df['weight'] > 200, 'weight'] = df.loc[df['weight'] > 200, 'weight'] / 1000

    # Calculate BMI where possible
    for df in [train, test]:
        mask = df['bmi'] > 80
        df.loc[mask,'bmi'] = df.loc[mask,'weight'] / (df.loc[mask,'height']/100)**2

    for df in [train, test]:
        mask = df['bmi'] < 10
        df.loc[mask,'bmi'] = df.loc[mask,'weight'] / (df.loc[mask,'height']/100)**2

    # If the BMI is still missing after this calculation, we can fill with the median
    imputer = SimpleImputer(strategy='median')
    train[['bmi']] = imputer.fit_transform(train[['bmi']])
    test[['bmi']] = imputer.transform(test[['bmi']])

    # Columns to check for missing values. For each column, create a new '_flag' column and fill missing values with mean
    columns_to_check = ['steps_day_2', 'test_2', 'test_6', 'test_8', 'test_10', 'test_12', 'test_15']
    for col in columns_to_check:
        train[col + '_flag'] = np.where(train[col].isna(), 0, 1)
        test[col + '_flag'] = np.where(test[col].isna(), 0, 1)

    train[columns_to_check] = imputer.fit_transform(train[columns_to_check])
    test[columns_to_check] = imputer.transform(test[columns_to_check])

    # Drop the 'country' column from train and test dataframes
    train.drop(['country'], axis=1, inplace=True)
    test.drop(['country'], axis=1, inplace=True)
    train.drop(['region'], axis=1, inplace=True)
    test.drop(['region'], axis=1, inplace=True)

    # Create a SimpleImputer object with strategy as 'most_frequent'. Replace missing values with "NaN"
    frequent_imputer = SimpleImputer(strategy='most_frequent')
    train['HMO'].fillna('NaN', inplace=True)
    test['HMO'].fillna('NaN', inplace=True)
    train['city'].fillna('NaN', inplace=True)
    test['city'].fillna('NaN', inplace=True)
    train['employment'].fillna('NaN', inplace=True)
    test['employment'].fillna('NaN', inplace=True)

    # Convert dates to datetime format and extract year, month, and day
    train['created_at'] = pd.to_datetime(train['created_at'])
    train['created_year'] = train['created_at'].dt.year
    train['created_month'] = train['created_at'].dt.month
    train['created_day'] = train['created_at'].dt.day
    train.drop('created_at', axis=1, inplace=True)
    train['birth_date'] = pd.to_datetime(train['birth_date'])
    test['created_at'] = pd.to_datetime(test['created_at'])
    test['created_year'] = test['created_at'].dt.year
    test['created_month'] = test['created_at'].dt.month
    test['created_day'] = test['created_at'].dt.day
    test.drop('created_at', axis=1, inplace=True)
    test['birth_date'] = pd.to_datetime(test['birth_date'])

    # Create a new feature for age (in years and fraction of year) from the birth_date and drop birth_date
    now = pd.to_datetime('today')
    train['age'] = (now.year - train['birth_date'].dt.year) + ((now.month - train['birth_date'].dt.month) / 12.0)
    train.drop('birth_date', axis=1, inplace=True)
    test['age'] = (now.year - test['birth_date'].dt.year) + ((now.month - test['birth_date'].dt.month) / 12.0)
    test.drop('birth_date', axis=1, inplace=True)

    # Separate target variable 'label' from the training data
    target = train['label']
    train.drop(['label', 'patient_id'], axis=1, inplace=True)

    # Store patient_ids of the test set for final submission, and drop from test set
    test_ids = test['patient_id']
    test.drop(['patient_id'], axis=1, inplace=True)

    # Encode categorical columns using LabelEncoder
    cat_cols = ['employment', 'HMO', 'city']
    label_encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        train[col] = le.fit_transform(train[col])
        label_encoders[col] = le
        test[col] = le.transform(test[col])

    final_pross = pd.concat([train, test], axis=0)
    final_pross.to_csv('final_procc.csv', index=False)

    return train, test, target, test_ids, label_encoders, pre
