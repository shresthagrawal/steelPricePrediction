
import numpy
import csv

data =[]
#dataset = numpy.loadtxt("raw.csv", delimiter=",")
with open('raw.csv') as csvfile:
     spamreader = csv.DictReader(csvfile)
     for row in spamreader:
         if row['Prices']!= '-':
             data.append(row)
         

print(data[0]['Prices'])
# split into input (X) and output (Y) variables
#X = dataset[:,0:6]
#Y = dataset[:,6]
