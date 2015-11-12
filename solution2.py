# all the data files should be put under the folder localData/
import csv

class DecisionTree():
    def calculateOutcome(self, row):
        outcome = 0

        # Decision tree from weka j48 algorithm on training data
        if self.stringToFloat(row['bids_per_auction_mean']) > 5.780105:
            if self.stringToFloat(row['ips_per_bidder_per_auction_mean']) <= 4.333333:
                if self.stringToInt(row['n_bids']) > 137:
                    if self.stringToInt(row['jewelry']) == 1:
                        if self.stringToInt(row['monday']) <= 28:
                            outcome = 1
                    else:
                        if row['sleep'] == 'TRUE':
                            if self.stringToInt(row['sporting goods']) == 0:
                                if self.stringToInt(row['n_urls']) <= 9:
                                    if self.stringToFloat(row['ip_entropy']) <= 30.814723:
                                        if self.stringToInt(row['wednesday']) <= 98:
                                            outcome = 1
                                    else:
                                        outcome = 1
                        else:
                            outcome = 1
            else:
                if self.stringToInt(row['max_bids_in_hour72']) > 16:
                    if row['sleep'] == 'TRUE':
                        if self.stringToInt(row['num_first_bid']) <= 0:
                            if self.stringToInt(row['on_ip_that_has_a_bot']) == 1:
                                if self.stringToInt(row['address_infrequent_address']) == 1:
                                    outcome = 1
                            else:
                                outcome = 1
                        else:
                            outcome = 1
                    else:
                        if self.stringToFloat(row['on_ip_that_has_a_bot_mean']) > 0.463704:
                            if self.stringToFloat(row['countries_per_bidder_per_auction_mean']) <= 1.333333:
                                if self.stringToFloat(row['only_one_user']) > 0.136364:
                                    outcome = 1
                            else:
                                outcome = 1

        return outcome

    def stringToFloat(self, value):
        return float(value) if value else 0.0

    def stringToInt(self, value):
        return int(value) if value else 0

if __name__ == "__main__":
    inputFilePath = 'localData/features_test.csv'
    outputFilePath = 'outcome_j48weka.csv'
    decisionTree = DecisionTree()

    with open(inputFilePath, newline='') as inputCsvFile:
        with open(outputFilePath, 'w+', newline='') as outputCsvFile:
            reader = csv.DictReader(inputCsvFile, delimiter=',', quotechar='|')

            outputFieldNames = ['bidder_id', 'prediction']
            writer = csv.DictWriter(outputCsvFile, delimiter=',', quotechar='|', fieldnames=outputFieldNames)
            writer.writeheader()

            for row in reader:
                if decisionTree.stringToInt(row['outcome']) == -1:
                    outcome = decisionTree.calculateOutcome(row)
                    writer.writerow({'bidder_id': row['bidder_id'], 'prediction': float(outcome)})