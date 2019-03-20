
import numpy
import csv
import pickle


def make_data(file_name):
    # main data cleaning
    data =[]
    with open(file_name) as csvfile:
        dict = csv.DictReader(csvfile)
        data = []
        for row in dict:
            include = True
            for key, value in row.items():
                if value=='' or value=='-':
                    include = False
            #Remove all the rows which has nul iron ore value
            if(row['Prices Domestic (Pellet) (in INR/MT)']=='0'):
                include = False
            if include == True:
                data.append(row)

    #convert dict values to list
    sec_data =[]
    for row in data:
        sec_data.append(list(row.values()))

    #convert 2d list into 2d numpy array
    np_data = numpy.array(sec_data)
    print(len(np_data))

    # remove date because verbose
    #npData = numpy.delete(npData,[7,13],axis=1)

    # open the file to save data (pickle)
    print(np_data)
    with open('data/Stats.pickle', 'wb') as handle:
        pickle.dump(np_data, handle)

if __name__== '__main__':
    make_data('data/Stats.csv')
