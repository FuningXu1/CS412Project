import pandas as pd
import numpy as np
import sklearn.preprocessing as preprocessing
from sklearn.ensemble import RandomForestClassifier

# This requires python 2.7 instead of python 3

def predict(file_path):
    # X = pd.read_csv('/Users/Xu/PycharmProjects/untitled/localData/features.csv')
    features = pd.read_csv(file_path + 'features.csv')
    # construct output data object
    output = features[features.outcome == -1]['bidder_id'].reset_index()

    '''
    data cleaning
    '''
    features.drop(features[features.outcome >= 0][features.n_bids.isnull()].index, inplace=True)
    features = features.sort(['dt_others_median'])
    features = features.fillna(method='pad')
    features = features.fillna(method='backfill')
    features.sort_index(inplace=True)
    features = features.fillna(0)

    features = features.drop('most_common_country', 1)
    features = features.drop('bidder_id', 1)

    '''
    build classifier
    '''
    classifier_list = []
    number_of_classifiers = 5
    for i in range(number_of_classifiers):
        classifier_list.append(RandomForestClassifier(n_estimators=800, criterion='entropy'))
        # classifier_list.append(RandomForestClassifier(n_estimators=800, min_samples_leaf=1, random_state=i, criterion='entropy'))

    # extract training outcome
    outcome = features['outcome'].values
    # convert True/False to 1/0
    feature_cols = 1.0 * features.drop('outcome', 1)

    '''
    data normalization
    '''
    feature_cols = preprocessing.normalize(feature_cols.values, axis=0)
    num_of_train_data = features[features.outcome >= 0].shape[0]

    # extract train feature
    train_feature = feature_cols[0:num_of_train_data, :]
    # extract test feature
    test_feature = feature_cols[num_of_train_data:, :]
    # extract train outcome
    train_outcome = outcome[0:num_of_train_data]

    predictions = np.zeros(test_feature.shape[0])
    for j in range(number_of_classifiers):
        classifier_list[j].fit(train_feature, train_outcome)
        a = classifier_list[j].predict_proba(test_feature)[:,1]
        predictions += a

    # average up prediction scores
    predictions = 1.0 * predictions / number_of_classifiers

    # output the csv file
    output['prediction'] = pd.Series(predictions, index=output.index)
    output.drop('index', 1)
    output.to_csv(file_path + 'prediction.csv', sep=',', index=False, header=True, columns=['bidder_id', 'prediction'])

if __name__ == "__main__":
    predict('/Users/Xu/PycharmProjects/untitled/localData/')