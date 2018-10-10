import pickle
from autokeras.supervised import Supervised


if __name__== '__main__':

    cleanData('raw.csv')
    dataset = getData('pickleFile')    
    
    # split into input (X) and output (Y) variables
    X = dataset[:,0:17]
    Y = dataset[:,17]
    print(dataset[0:50,:])