# all the data files should be put under the folder localData/
import csv
import numpy as np
from sklearn import linear_model, datasets


class FileReader():
    def read_file(self, filename):
        with open(filename, newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=',', quotechar='|')
            for row in reader:
                print(row)

if __name__ == "__main__":
    fileReader = FileReader()
    fileReader.read_file('localData/test.csv')
