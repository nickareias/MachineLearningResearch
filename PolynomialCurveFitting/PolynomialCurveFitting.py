import numpy as np
import matplotlib.pyplot as plt
import sympy
import scipy.optimize as optimize

data_size = 1/10

x_vals = [x*data_size for x in range(int(1/data_size))]

#create synthetic dataset based on sin(2*pi*x) with added gaussian random noise
#with standard deviation of 0.3
y_vals = [np.sin(2*np.pi*x) + (np.random.randn(1)[0] * 0.3) for x in x_vals]

#plot synthetic data
plt.plot(x_vals, y_vals)


#create vector of coefficients
order = 2
w = [np.random.randn(1)[0] for x in range(order + 1)]


#plot fitted function
data_size_curve = 1/100
x_vals_curve = np.arange(0,1,data_size_curve)

y_vals_curve = [fn(x, w) for x in x_vals_curve]

plt.plot(x_vals_curve, y_vals_curve)
plt.show()
plt.clf()

#function of form w0*x^0 + w1*x^1 + w2*x^2 ... +wn*x^n
#x is a single value, w is a vector
def fn(x, w):
    summation = 0
    for (i, d) in enumerate(w):
        summation += d*np.power(x,i)
        
    return summation

#x values, target y values, w = coefficients
#all inputs are vectors
def error(X):
    
    x_vals = X[0]
    y_vals = X[1]
    w = X[2]
    e = 0
    for (x, y) in zip(x_vals, y_vals):
        e += np.square(fn(x,w) - y)
        
    e *= 0.5
    return e