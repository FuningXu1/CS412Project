# all the data files should be put under the folder localData/
import csv
import numpy as np
from sklearn import linear_model, datasets
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier


class FileReader():
    def read_file(self, filename):
        with open(filename, newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=',', quotechar='|')
            is_header = True
            ids = []
            result = []
            outcome = []
            for row in reader:
                if is_header:
                    is_header = False
                else:
                    ids.append(row[1])
                    # result.append((self.process_row_with_selected_feature(row)))
                    # result.append((self.process_row_with_self_selected_feature(row)))
                    result.append((self.process_row(row)))
                    if filename == 'localData/features_train.csv':
                        outcome.append(int(row[2]))
        return (ids, result, outcome)

    # 180 the median time between a user's bid and that user's previous bid
    # 181 the mean number of bids a user made per auction
    # the entropy for how many bids a user placed on each day of the week
    # 78 the means of the per-auction URL entropy and IP entropy for each user
    # 164 the maximum number of bids in a 20 min span
    # 182 the total number of bids placed by the user
    # 184 the average number of bids a user placed per referring URL
    # the number of bids placed by the user on each of the three weekdays in the data
    # the minimum and median times between a user's bid and the previous bid by another user in the same auction.
    # 83 the fraction of IPs used by a bidder which were also used by another user which was a bot

    def process_row_with_selected_feature(self, row):
        result = []
        for index, col in enumerate(row):
            if index in (180, 181, 78, 164, 182, 184, 83):
                if col == '':
                    result.append(0.0)
                elif col == 'TRUE':
                    result.append(1.0)
                elif col == 'FALSE':
                    result.append(0.0)
                else:
                    result.append(float(col))
        return result

    # 78 mean bid per auctions entropy
    # 80 ip for bidders
    # 83 84 isBot(Bool) mean value
    # 88 num per bid
    # 90 91 time until first bid
    # 164 max_bids_in_hour 72
    # 165 sleep
    # 181 bids per auction mean
    # 200 countries_per_bidder_per_auction

    def process_row_with_self_selected_feature(self, row):
        result = []
        for index, col in enumerate(row):
            if index in (78, 80, 83, 84, 88, 90, 91, 164, 165, 181, 200):
                if col == '':
                    result.append(0.0)
                elif col == 'TRUE':
                    result.append(1.0)
                elif col == 'FALSE':
                    result.append(0.0)
                else:
                    result.append(float(col))
        return result

    def process_row(self, row):
        result = []
        for index, col in enumerate(row):
            if index >= 3 and index != 202:
                if col == '':
                    result.append(0.0)
                elif col == 'TRUE':
                    result.append(1.0)
                elif col == 'FALSE':
                    result.append(0.0)
                else:
                    result.append(float(col))
        return result

class Classifer():
    def gaussianNB(self, train_rows, train_outcome, test_rows):
        clf = GaussianNB()
        clf.fit(train_rows, train_outcome)
        return clf.predict(test_rows)

    def randomForestClassifier(self, train_rows, train_outcome, test_rows):
        # n_estimators=800, max_depth=None, min_samples_leaf=1, random_state=i, criterion='entropy'
        clf = RandomForestClassifier(n_estimators=800, max_depth=None, min_samples_leaf=1, criterion='entropy')
        # clf = RandomForestClassifier(n_estimators=5)
        clf.fit(train_rows, train_outcome)
        return clf.predict(test_rows)

    def kNeighborsClassifier(self, train_rows, train_outcome, test_rows):
        neigh = KNeighborsClassifier(n_neighbors=3)
        neigh.fit(train_rows, train_outcome)
        return neigh.predict(test_rows)

if __name__ == "__main__":
    fileReader = FileReader()
    (train_ids, train_rows, train_outcome) = fileReader.read_file('localData/features_train.csv')
    (test_ids, test_rows, test_outcome) = fileReader.read_file('localData/features_test.csv')

    classfier = Classifer()
    # predictions = classfier.gaussianNB(train_rows, train_outcome, test_rows)
    predictions = classfier.randomForestClassifier(train_rows, train_outcome, test_rows)
    # predictions = classfier.kNeighborsClassifier(train_rows, train_outcome, test_rows)


    # write to file
    with open('outcome.csv', 'w+') as f:
        f.write('bidder_id,prediction\n')
        count = 0
        for index, prediction in enumerate(predictions):
            f.write(test_ids[index] + ',' + str(float(prediction)) + '\n')



