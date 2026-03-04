import csv

def clearDataFile(fileName):
    open(fileName, 'w').close()

clearDataFile('backtestingData.csv')