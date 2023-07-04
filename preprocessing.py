import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from scipy import stats
import numpy as np

def preprocess_data(personal_info_train, measurements_train, personal_info_test, measurements_test):
    
    train = pd.merge(personal_info_train, measurements_train, on='patient_id')
    test = pd.merge(personal_info_test, measurements_test, on='patient_id')

    # Remove duplicates
    train.drop_duplicates(subset='patient_id', keep='first', inplace=True)
   
    # Drop unnecessary columns
    train.drop(['country', 'region'], axis=1, inplace=True)
    test.drop(['country', 'region'], axis=1, inplace=True)

    train_stats_df = pd.DataFrame()
    
    for column in train.columns:
        # Basic statistics
        column_stats = train[column].describe()
        
        # Additional statistics
        column_stats['missing_values'] = train[column].isna().sum()
        column_stats['missing_percentage'] = (train[column].isna().sum() / train.shape[0]) * 100
        column_stats['unique_values'] = train[column].nunique()        
        # Concatenate statistics to main dataframe
        train_stats_df = pd.concat([train_stats_df, column_stats], axis=1)

    # Transpose the DataFrame for a more intuitive format and save to CSV
    train_stats_df.transpose().to_csv('train_statistics.csv')
    print("file saved as train_statistics.csv")

    test_stats_df = pd.DataFrame()

    for column in test.columns:
        # Basic statistics
        column_stats = test[column].describe()
        
        # Additional statistics
        column_stats['missing_values'] = test[column].isna().sum()
        column_stats['missing_percentage'] = (test[column].isna().sum() / test.shape[0]) * 100
        column_stats['unique_values'] = test[column].nunique()        
        # Concatenate statistics to main dataframe
        test_stats_df = pd.concat([test_stats_df, column_stats], axis=1)

    # Transpose the DataFrame for a more intuitive format and save to CSV
    test_stats_df.transpose().to_csv('test_statistics.csv')
    print("file saved as test_statistics.csv")

    # Calculate the number of missing values for each row
    train['missing_count'] = train.isnull().sum(axis=1)

    # Set a threshold
    threshold = 4

    # Calculate how many rows have more than 4 missing values
    num_dropped_train = sum(train['missing_count'] > threshold)

    # Print how many rows will be dropped from each dataset
    print(f"Number of rows dropped from train: {num_dropped_train}, they had more than {threshold} missing values")

    # Drop rows in train and test data that have missing_count greater than the threshold
    train = train[train['missing_count'] <= threshold]

    # Now you can drop the 'missing_count' column as it's no longer needed
    train.drop('missing_count', axis=1, inplace=True)


    # Map gender to numeric values
    # Fix height and weight outliers
    # Calculate BMI where possible
    # Fill missing BMIs with median
    gender_map = {'M': 0, 'F': 1}
    train['gender'] = train['gender'].map(gender_map)
    test['gender'] = test['gender'].map(gender_map)
    for df in [train, test]:
        df.loc[df['height'] < 10, 'height'] = df.loc[df['height'] < 10, 'height'] * 100
        df.loc[df['weight'] > 200, 'weight'] = df.loc[df['weight'] > 200, 'weight'] / 1000
    for df in [train, test]:
        mask = df['bmi'] > 80
        df.loc[mask,'bmi'] = df.loc[mask,'weight'] / (df.loc[mask,'height']/100)**2
        mask = df['bmi'] < 10
        df.loc[mask,'bmi'] = df.loc[mask,'weight'] / (df.loc[mask,'height']/100)**2
    imputer = SimpleImputer(strategy='median')
    #imputer = IterativeImputer(max_iter=10, random_state=0)

    #train[['bmi']] = imputer.fit_transform(train[['bmi']])
    #test[['bmi']] = imputer.transform(test[['bmi']])

    # Fill missing values and add flags
    columns_to_check = ['steps_day_2', 'test_2', 'test_6', 'test_8', 'test_10', 'test_12', 'test_15']
    for col in columns_to_check:
         train[col + '_flag'] = np.where(train[col].isna(), 0, 1)
         test[col + '_flag'] = np.where(test[col].isna(), 0, 1)
    imputer = KNNImputer(n_neighbors=20, weights='uniform', metric='nan_euclidean')
    train[columns_to_check] = imputer.fit_transform(train[columns_to_check])
    test[columns_to_check] = imputer.transform(test[columns_to_check])


    # Calculate the average of 'steps_day_1', 'steps_day_3', 'steps_day_4', 'steps_day_5' for train and test dataframes
    # Fill missing values in 'steps_day_2' with the calculated average for train and test dataframes
    train['avg_steps'] = train[['steps_day_1', 'steps_day_3', 'steps_day_4', 'steps_day_5']].mean(axis=1)
    test['avg_steps'] = test[['steps_day_1', 'steps_day_3', 'steps_day_4', 'steps_day_5']].mean(axis=1)
    train['steps_day_2'].fillna(train['avg_steps'], inplace=True)
    test['steps_day_2'].fillna(test['avg_steps'], inplace=True)
    train.drop('avg_steps', axis=1, inplace=True)
    test.drop('avg_steps', axis=1, inplace=True)

    # Fill missing categorical columns
    train.loc[:, ['HMO', 'city', 'employment']] = train[['HMO', 'city', 'employment']].fillna('NaN')
    test.loc[:, ['HMO', 'city', 'employment']] = test[['HMO', 'city', 'employment']].fillna('NaN')

    #Convert dates to datetime format and extract year, month, and day
    #Create a new feature for age (in years and fraction of year) from the birth_date and drop birth_date
    # train['created_at'] = pd.to_datetime(train['created_at'])
    # train['created_year'] = train['created_at'].dt.year + (train['created_at'].dt.month/12)
    # train.drop('created_at', axis=1, inplace=True)
    # train['birth_date'] = pd.to_datetime(train['birth_date'])
    # test['created_at'] = pd.to_datetime(test['created_at'])
    # test['created_year'] = test['created_at'].dt.year + (test['created_at'].dt.month/12)
    # test.drop('created_at', axis=1, inplace=True)
    # test['birth_date'] = pd.to_datetime(test['birth_date'])
    # now = pd.to_datetime('today')
    # train['age'] = (now.year - train['birth_date'].dt.year) + ((now.month - train['birth_date'].dt.month) / 12.0)
    # train.drop('birth_date', axis=1, inplace=True)
    # test['age'] = (now.year - test['birth_date'].dt.year) + ((now.month - test['birth_date'].dt.month) / 12.0)
    # test.drop('birth_date', axis=1, inplace=True)

    train['created_at'] = pd.to_datetime(train['created_at'])
    train['created_year'] = train['created_at'].dt.year + (train['created_at'].dt.month/12)
    train['birth_date'] = pd.to_datetime(train['birth_date'])
    train['age'] = (train['created_at'].dt.year - train['birth_date'].dt.year) + ((train['created_at'].dt.month - train['birth_date'].dt.month) / 12.0)
    train.drop(['created_at', 'birth_date'], axis=1, inplace=True)

    test['created_at'] = pd.to_datetime(test['created_at'])
    test['created_year'] = test['created_at'].dt.year + (test['created_at'].dt.month/12)
    test['birth_date'] = pd.to_datetime(test['birth_date'])
    test['age'] = (test['created_at'].dt.year - test['birth_date'].dt.year) + ((test['created_at'].dt.month - test['birth_date'].dt.month) / 12.0)
    test.drop(['created_at', 'birth_date'], axis=1, inplace=True)


    # Separate target variable
    target = train['label']
    train.drop(['label', 'patient_id'], axis=1, inplace=True)

    # Store patient_ids for final submission
    test_ids = test['patient_id']
    test.drop(['patient_id'], axis=1, inplace=True)

    # Encode categorical columns
    cat_cols = ['employment', 'HMO', 'city']
    label_encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        le.fit(train[col])
        train[col] = le.transform(train[col])
        label_encoders[col] = le
        try:
            test[col] = le.transform(test[col])
        except Exception as e:
            print(f"Error occurred while encoding {col} in test dataset: {e}")

    print('Data processed')

    # Save processed data
    final_pross = pd.concat([train, test], axis=0)
    final_pross.to_csv('final_procc.csv', index=False)
    print('final_procc.csv saved')

    return train, test, target, test_ids, cat_cols

