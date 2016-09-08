import numpy as np


def data():
    array = np.zeros(19)
    operation_array = [1,0]
    new_array = []
    for i in xrange(1,10):
        for j in xrange(10,19):
            if i >= (j-9):
                for k in xrange(2):
                    array[0:i] = 1
                    array[9:j] = 1
                    array[18] = operation_array[k]
                    new_array.append(array)
                    array = np.zeros(19)
            if i < (j-9):
                array[0:i] = 1
                array[9:j] = 1
                array[18] = operation_array[0]
                new_array.append(array)
                array = np.zeros(19)
    return(new_array)

data()
#print(new_array)
#print(len(new_array))
