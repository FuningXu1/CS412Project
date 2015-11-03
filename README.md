# CS412Project

Online csv-arff converter: http://ikuz.eu/csv2arff/

Or if you can run java, you can do the following in the command line:
java -cp /path/to/weka.jar weka.core.converters.CSVLoader filename.csv > filename.arff

I was having issues when converting to arff, apparently it was because towards the end, some rows have empty values in the last few columns. I just replace those missing values with 0s, but not sure if that's the right way to do it.
