import numpy as np
data = np.genfromtxt(r'DataSet(Type1).csv',delimiter = '|' ,names = True, dtype = None,)


xinput = np.zeros((126,19))
yinput = np.zeros((126,18))

for i in xrange (126):
    for j in xrange(1,38,2):
        xinput[i,round(j/2)] = float(data[i][0][j])
    for k in xrange(41,76,2):
        yinput[i,round((k-41)/2)] = float(data[i][0][k])


print type(float(data[0][0][1]))
