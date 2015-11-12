# all the data files should be put under the folder localData/
import csv
from sklearn.naive_bayes import GaussianNB, BernoulliNB
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
                    result.append((self.process_row_with_selected_features(row)))
                    #result.append((self.process_row(row)))
                    if filename == 'localData/features_train.csv':
                        outcome.append(int(row[2]))
        return (ids, result, outcome)

    def process_row_with_selected_features(self, row):
        result = []
        for index, col in enumerate(row):
            if index in (3, 56, 57, 199, 200, 201) or (index >= 75 and index <= 91) or (index >= 164 and index <= 185) or index >= 402:
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

class Classifier():
    def gaussianNB(self, train_rows, train_outcome, test_rows):
        clf = GaussianNB()
        clf.fit(train_rows, train_outcome)
        return clf.predict(test_rows)

    def bernoulliNB(self, train_rows, train_outcome, test_rows):
        clf = BernoulliNB()
        clf.fit(train_rows, train_outcome)
        return clf.predict(test_rows)

    def randomForestClassifier(self, train_rows, train_outcome, test_rows):
        clf = RandomForestClassifier(n_estimators=800, max_depth=None, min_samples_leaf=1, criterion='entropy')
        clf.fit(train_rows, train_outcome)
        return clf.predict(test_rows)

    def kNeighborsClassifier(self, train_rows, train_outcome, test_rows):
        neigh = KNeighborsClassifier(n_neighbors=100)
        neigh.fit(train_rows, train_outcome)
        return neigh.predict(test_rows)

if __name__ == "__main__":
    fileReader = FileReader()
    (train_ids, train_rows, train_outcome) = fileReader.read_file('localData/features_train.csv')
    (test_ids, test_rows, test_outcome) = fileReader.read_file('localData/features_test.csv')

    classifier = Classifier()
    gnbPredictions = classifier.gaussianNB(train_rows, train_outcome, test_rows)
    bnbPredictions = classifier.bernoulliNB(train_rows, train_outcome, test_rows)
    rfcPredictions = classifier.randomForestClassifier(train_rows, train_outcome, test_rows)
    kncPredictions = classifier.kNeighborsClassifier(train_rows, train_outcome, test_rows)

    # write to file
    with open('outcome_gaussian_naive_bayes.csv', 'w+') as f:
        f.write('bidder_id,prediction\n')
        count = 0
        for index, prediction in enumerate(gnbPredictions):
            f.write(test_ids[index] + ',' + str(float(prediction)) + '\n')

    with open('outcome_bernoulli_naive_bayes.csv', 'w+') as f:
        f.write('bidder_id,prediction\n')
        count = 0
        for index, prediction in enumerate(bnbPredictions):
            f.write(test_ids[index] + ',' + str(float(prediction)) + '\n')

    with open('outcome_random_forest.csv', 'w+') as f:
        f.write('bidder_id,prediction\n')
        count = 0
        for index, prediction in enumerate(rfcPredictions):
            f.write(test_ids[index] + ',' + str(float(prediction)) + '\n')

    with open('outcome_k_neighbors.csv', 'w+') as f:
        f.write('bidder_id,prediction\n')
        count = 0
        for index, prediction in enumerate(kncPredictions):
            f.write(test_ids[index] + ',' + str(float(prediction)) + '\n')