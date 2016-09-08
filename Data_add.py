import numpy as np

def data_add():
    array = np.zeros(18)
    array2 = np.zeros(18)
    new_array, new_array2 = [],[]
    for i in xrange(1,10):
        for j in xrange(10,19):
            #print(i,j)
            k = i+(j-10)
            array[i-1] = 1
            array[j-1] = 1
            array2[k] = 1
            new_array2.append(array2)
            new_array.append(array)
            array = np.zeros(18)
            array2 = np.zeros(18)
            
    #print new_array
    return (new_array, new_array2)

print len(data_add()[0])
