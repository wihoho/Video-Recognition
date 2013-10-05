import numpy as np

x = [0.5] * 10
x = np.array(x).reshape(1, 10)

y = [2] * 10
y = np.array(y).reshape(10,1)

z = x * y

print "Yes"
