# Lib Imports
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression, LassoCV
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

# Setting Pandas column display option
pd.set_option('display.max_columns', 500)


def tar_feat(df):
    target = ['isFraud']
    to_remove = ['isFraud', 'TransactionID']
    features = df.columns.to_list()
    features = [x for x in features if not x in to_remove]
    return target, features


def pipeline(idee, transaction):
    merge = transaction.merge(idee, how='outer', on='TransactionID')
    objects = merge.select_dtypes('object')
    objects.fillna("Unknown", inplace=True)
    objects = pd.get_dummies(objects)
    objects['TransactionID'] = merge['TransactionID']
    cols = objects.columns.to_list()
    cols = cols[-1:] + cols[:-1]
    objects = objects[cols]
    objects.astype('int64')
    numbers = merge.select_dtypes(include=['float64', 'int64'])
    numbers.fillna(numbers.mean(), inplace=True)
    df = objects.merge(numbers, how='outer', on='TransactionID')
    return df


def fraud_uds(df):
    total = df.shape[0]
    counts = df['isFraud'].value_counts()
    no_fraud = counts[0]
    fraud = counts[1]
    fraud_df = df[df['isFraud'] == 1]
    no_fraud_df = df[df['isFraud'] == 0]
    no_fraud_df = no_fraud_df.iloc[:fraud]
    undersample = pd.concat([no_fraud_df, fraud_df])
    return undersample

def random_uds(X,y):
    rusampler = RandomUnderSampler()
    X_rus, y_rus = rusampler.fit_resample(X, y)
    return X_rus, y_rus

if __name__ == '__main__':
    # idee = pd.concat([idee_train, idee_test])
    # transaction = pd.concat([transaction_train, transaction_test])

    idee_train = pd.read_csv('train_identity.csv')
    transaction_train = pd.read_csv('train_transaction.csv')

    print('Initiating Pipeline Processing on Training Dataset')
    df = pipeline(idee_train, transaction_train)
    print()
    print('Undersampling')
    df = fraud_uds(df)
    print()
    print('Separate Target from Features data')
    target, features = tar_feat(df)
    X = df[features]
    y = df[target]
    
    print()
    print('Initiating Data Splitting')
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=123)
    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

    print()
    print('Instantiating and Training Model')
    model = LogisticRegression(
        penalty='l2', C=1e42, max_iter=150, verbose=1, solver='liblinear', n_jobs=-1)
    model.fit(X_train, y_train)
    test_score = model.score(X_test, y_test)
    y_pred = model.predict(X_test)
    print()
    print()
    print('Score Result:')
    print(str(float(test_score)*100))
    print()
    print('Classification Report:')
    print()
    print(classification_report(y_test, y_pred))
    
    # idee_test = pd.read_csv('test_identity.csv')
    # transaction_test = pd.read_csv('test_transaction.csv')
    # df = pipeline(idee_test, transaction_test)
    # df = df[set(df.columns.to_list()).intersection(X_test.columns.to_list())]
    # model.predict(df)
    
