
import numpy


dataset = numpy.loadtxt("raw.csv", delimiter=",")
# split into input (X) and output (Y) variables
X = dataset[:,0:6]
Y = dataset[:,6]
print(dataset)
