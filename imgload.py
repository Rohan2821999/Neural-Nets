from PIL import Image
import numpy as np


def ImgCon():
    list_images = ["1_1","1_2","1_3","2_1","2_2","2_3","3_1","3_2","3_3","4_1","4_2","4_3","5_1","5_2","5_3","6_1","6_2","6_3","7_1","7_2","7_3","8_1","8_2","8_3","9_1","9_2","9_3"]
    final_array = []
    #print(list_images[0] + '.png')


    for k in range(len(list_images)):
        new_arr = []
        img = Image.open(list_images[k] + '.png').convert("L")

        array = np.array(img)
        array[array == 255] = 1
        array = np.resize(array,(2500,1))

        for i in xrange(len(array)):
            new_arr.append(int(array[i]))
        final_array.append(new_arr)
    print(len(final_array))
    return(final_array)




ImgCon()
