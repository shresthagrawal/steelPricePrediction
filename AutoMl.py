
import numpy
import csv

#function to tokenize words
dict = {'-':0}

def tokenize(arrStr):
    res = []
    for str in arrStr:
        if str in dict:
            res.append(dict[str])
        else:
            dict.update({str:len(dict)})
            res.append(dict[str])
    return res
    
if __name__== '__main__':
    # main data cleaning 
    data =[]
    with open('raw.csv') as csvfile:
         spamreader = csv.DictReader(csvfile)
         for row in spamreader:
             # remove all the values which has nul iron ore value
             if row['Prices']!= '-':
                 data.append(row)

    print(data[0].keys())

    #convert dict values to list 
    secData =[]
    for x in data:
        secData.append(x.values())

    #convert 2d list into 2d numpy array
    npData = numpy.array(secData)

    # remove date because verbose
    #np.delete(arr, [0,2,4], axis=0)
    npData = numpy.delete(npData,[7,13],axis=1)

    #tokenize data to convert word data to int
    for y in [0,1,2,3,5,6,7,8,10,11,13,14,15,16,17] :
        npData[:,y]=tokenize(npData[:,y])
        
    #npData[:,0]= tokenize(npData[:,0])
    print(npData[0:50,:])
    
    # split into input (X) and output (Y) variables
    #X = dataset[:,0:6]
    #Y = dataset[:,6]
