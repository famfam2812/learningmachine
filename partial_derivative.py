import matplotlib
import nnfs
import numpy as np

#forward pass
x = [1.0, -2.0, 3.0] #input
w = [-3.0, -1.0, 2.0] #weights
b = 1.0 # bias

#multiplying inputs by weights
wx0 = x[0] * w[0]
wx1 = x[1] * w[1]
wx2 = x[2] * w[2]

#adding
s = wx0 + wx1 + wx2 + b


#ReLU activation function
y = max(s, 0)
print(y)


dy = (1 if s > 0 else 0)

dwx0 = 1 * dy
dwx1 = 1 * dy
dwx2 = 1 * dy
db = 1 * dy

dx0 = w[0] * dwx0
dw0 = x[0] * dwx0

dx1 = w[1] * dwx1
dw1 = x[1] * dwx1

dx2 = w[2] * dwx2
dw2 = x[2] * dwx2

dx = [dx0, dx1, dx2] # gradients on inputs
dw = [dw0, dw1, dw2] # gradients on weights
db # gradient on bias...just 1 bias here.

print(db, dx0, dw0, dx1, dw1, dx2, dw2)

w[0] += -0.001 * dw[0]
w[1] += -0.001 * dw[1]
w[2] += -0.001 * dw[2]
b += -0.001 * db

print(w, b)

# Multiplying inputs by weights
wx0 = x[0] * w[0]
wx1 = x[1] * w[1]
wx2 = x[2] * w[2]

# Adding
s = wx0 + wx1 + wx2 + b
# ReLU activation function
y = max(s, 0)
print(y)
