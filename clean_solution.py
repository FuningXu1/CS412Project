import pandas as pd
import numpy as np
from collections import Counter
import sklearn.preprocessing as preprocessing
from sklearn.tree import DecisionTreeClassifier # base estimator
from sklearn.base import clone


# This requires python 2.7 instead of python 3
'''
data processing
columns to keep:
'bidder_id', 'outcome',

selected features:

'payment_account_prefix_same_as_address_prefix', 'ips_per_bidder_per_auction_median',
'ips_per_bidder_per_auction_mean', 'ip_only_one_user_counts',
'on_ip_that_has_a_bot', 'on_ip_that_has_a_bot_mean',
'ip_entropy', 'dt_change_ip_median', 'dt_same_ip_median', 'num_first_bid',

'bids_per_auction_per_ip_entropy_median', 'bids_per_auction_per_ip_entropy_mean',
'ips_per_bidder_per_auction_median', 'ips_per_bidder_per_auction_mean',
'bids_per_auction_median', 'bids_per_auction_mean',
'countries_per_bidder_per_auction_median', 'countries_per_bidder_per_auction_mean',

'n_bids', 'n_bids_url',

'n_urls', 'f_urls', 'url_entropy',

'countries_per_bidder_per_auction_median', 'countries_per_bidder_per_auction_mean', 'countries_per_bidder_per_auction_max',

'address_rare_address', 'address_infrequent_address',

'payment_account_rare_account', 'payment_account_infrequent_account', 'only_one_user'


'''


def data_cleaning(file_path):
    features = pd.read_csv(file_path + 'features.csv')
    # construct output data object
    output = features[features.outcome == -1]['bidder_id'].reset_index()

    features = features[['bidder_id', 'outcome',
                         'payment_account_prefix_same_as_address_prefix', 'ips_per_bidder_per_auction_median',
    'ips_per_bidder_per_auction_mean', 'ip_only_one_user_counts',
    'on_ip_that_has_a_bot', 'on_ip_that_has_a_bot_mean',
    'ip_entropy', 'dt_change_ip_median', 'dt_same_ip_median', 'num_first_bid',
                             'bids_per_auction_per_ip_entropy_median', 'bids_per_auction_per_ip_entropy_mean',
    'ips_per_bidder_per_auction_median', 'ips_per_bidder_per_auction_mean',
    'bids_per_auction_median', 'bids_per_auction_mean',
    'countries_per_bidder_per_auction_median', 'countries_per_bidder_per_auction_mean',

    'n_bids', 'n_bids_url',

    'n_urls', 'f_urls', 'url_entropy',

    'countries_per_bidder_per_auction_median', 'countries_per_bidder_per_auction_mean', 'countries_per_bidder_per_auction_max',

    'address_rare_address', 'address_infrequent_address',

    'payment_account_rare_account', 'payment_account_infrequent_account', 'only_one_user'
                         ]]

    '''
    data cleaning
    '''
    features.drop(features[features.outcome >= 0][features.n_bids.isnull()].index, inplace=True)
    features = features.fillna(method='pad')
    features = features.fillna(method='backfill')
    features = features.fillna(0)

    features = features.drop('most_common_country', 1)
    features = features.drop('bidder_id', 1)

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

    return train_feature, train_outcome, test_feature, output


class RandomForestClassifierTest:

    def __init__(self, n_estimators=10):
        self.n_estimators = n_estimators
        self.criterion = "gini"
        # construct base estimator as decision tree classifer
        self.base_estimator = DecisionTreeClassifier(criterion=self.criterion)
        self.estimator_params=()

    # function to calculate entropy
    def entropy(self, data):
        value = Counter([b for _, b in data]).values()
        sum_value = float(sum(value))
        d = np.array(value) / sum_value
        return - sum(d * np.log(d))

    # function to fit the training data
    def fit(self, X, y):
        '''
        X : training data feature matrix
        y : outcome column
        '''

        n_samples, self.n_features_ = X.shape
        y = np.atleast_1d(y)
        if y.ndim == 1:
            y = np.reshape(y, (-1, 1))

        trees = []
        random_seed = np.random.mtrand._rand # get a random seed
        for i in range(self.n_estimators):
            tree = self.get_base_classifer()
            tree.set_params(random_state=random_seed.randint(np.iinfo(np.int32).max))
            trees.append(tree)

        trees = [self.grow_tree(t, X, y) for t in trees]
        self.estimators.extend(trees)

        return self

    # grow trees
    def grow_tree(self, tree, X, y):
        n_samples = X.shape[0]
        weight = np.ones((n_samples,), dtype=np.float64)
        random_state = np.random.mtrand._rand
        indices = random_state.randint(0, n_samples, n_samples)
        sample_counts = np.bincount(indices, minlength=n_samples)
        weight *= sample_counts

        # call base estimator
        tree.fit(X, y, sample_weight=weight, check_input=False)
        return tree

    def predict_proba(self, X):
        '''
        X : test data feature matrix
        '''
        all_proba = []
        for tree in self.estimators:
            all_proba.append(tree.predict_proba(X))

        proba = all_proba[0]
        for j in range(1, len(all_proba)):
            proba += all_proba[j]

        proba /= len(self.estimators)
        return proba


    # helper function
    def get_base_classifer(self):

        self.estimators = []
        self.base_estimator_ = self.base_estimator

        estimator = clone(self.base_estimator)
        estimator.set_params(**dict((p, getattr(self, p))
                                    for p in self.estimator_params))

        return estimator

def predict(num_of_classifers, classifer, train_feature, train_outcome, test_feature, output):
    number_of_classifiers = num_of_classifers
    classifier_list = []
    for i in range(number_of_classifiers):
        classifier_list.append(classifer)

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
    file_path = '/Users/Xu/PycharmProjects/untitled/localData/'
    train_feature, train_outcome, test_feature, output = data_cleaning(file_path)
    predict(5, RandomForestClassifierTest(n_estimators=100), train_feature, train_outcome, test_feature, output)
