
from itertools import permutations
import numpy as np



def data3_add():
    array = np.zeros(9)
    new_array,store_array,x_array,y_labels_array  = [],[],[],[]
    y_labels = np.zeros(18)
    for i in xrange(1,10):
        array[0:i] = 1
        store_array.append(array)
        array = np.zeros(9)

    for i in xrange(len(store_array)):
        test_array = store_array[i]
        l = list(permutations(test_array))
        new_list = list(set(l))
    #print new_list
        for i in xrange(len(new_list)):
            new_array.append(list(new_list[i]))


    new_array2 = new_array



    for i in xrange(len(new_array)):
        for j in xrange(len(new_array2)):
            x_array.append(new_array[i]+new_array2[j])
            val = new_array[i].count(1) + new_array2[j].count(1)

            y_labels[val-1] = 1
            y_labels_array.append(y_labels)
            y_labels = np.zeros(18)

    print len(x_array)        
    return (x_array,y_labels_array)

data3_add()
