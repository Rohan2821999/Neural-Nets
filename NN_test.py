import numpy as np
import math

def Sigmoid(x):
    return 1/ (1+ math.exp(-x))

X_vals = [1,1]
vals,vals2 = [],[]
hidden_vals, S_vals, new_random_vals,mod_vals = [],[],[],[]

for i in xrange(3):
    vals.append(np.random.uniform(0,1))
    vals2.append(np.random.uniform(0,1))
    hidden_vals.append((X_vals[0]*vals[i])+ (X_vals[1]*vals2[i]))
    S_vals.append(Sigmoid(hidden_vals[i]))
    new_random_vals.append(np.random.uniform(0,1))
    mod_vals.append(S_vals[i]*new_random_vals[i])

print(sum(mod_vals))

