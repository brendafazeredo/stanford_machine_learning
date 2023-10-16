import numpy as np
import time

a = np.zeros(4)
print(f"np.zeros(4) :   a = {a}, a shape = {a.shape}, a data type = {a.dtype}")

a = np.zeros((4,))
print(f"np.zeros(4,) :  a = {a}, a shape = {a.shape}, a data type = {a.dtype}")

a = np.random.random_sample(4)
print(f"np.random.random_sample(4): a = {a}, a shape = {a.shape}, a data type = {a.dtype}")

a = np.arange(4.)
print(f"np.arrange(4.):     a = {a}, a shape = {a.shape}, a data type = {a.dtype}")

a = np.random.rand(4)
print(f"np.random.rand(4): a = {a}, a shape = {a.shape}, a data type = {a.dtype}")

a = np.array([5, 4, 3, 2])
print(f"np.array([5,4,3,2]):  a = {a},     a shape = {a.shape}, a data type = {a.dtype}")

a = np.array([5., 4, 3, 2])
print(f"np.array([5.,4,3,2]): a = {a}, a shape = {a.shape}, a data type = {a.dtype}")

a = np.arange(10)
print(a)

print(f"a[2].shape: {a[2].shape} a[2]  = {a[2]}, Accessing an element returns a scalar")

print(f"a[-1] = {a[-1]}")

try:
    c = a[10]
except Exception as e:
    print("The error message you'll see is:")
    print(e)

a = np.arange(10)
print(f"a         = {a}")

c = a[2:7:1]
print("a[2:7:1] = ", c)

c = a[2:7:2]
print("a[2:7:2] = ", c)

c = a[3:]
print("a[3:]    = ", c)

c = a[:3]
print("a[:3]    = ", c)

c = a[:]
print("a[:]     = ", c)


a = np.array([1, 2, 3, 4])
print(f"a             : {a}")
b = -a
print(f"b = -a        : {b}")

b = np.sum(a)
print(f"b = np.sum(a) : {b}")

b = np.mean(a)
print(f"b = np.mean(a): {b}")

b = a**2
print(f"b = a**2      : {b}")


a = np.array([1, 2, 3, 4])
b = np.array([-1, -2, 3, 4])
print(f"Binary operators work element wise: {a + b}")

c = np.array([1, 2])
try:
    d = a + c
except Exception as e:
    print("The error message you'll see is:")
    print(e)

a = np.array([1, 2, 3, 4])

b = 5 * a
print(f"b = 5 * a : {b}")


def my_dot(a, b): 
    """
   Compute the dot product of two vectors
 
    Args:
      a (ndarray (n,)):  input vector 
      b (ndarray (n,)):  input vector with same dimension as a
    
    Returns:
      x (scalar): 
    """
    x = 0
    for i in range(a.shape[0]):
        x = x + a[i] * b[i]
    return x


a = np.array([1, 2, 3, 4])
b = np.array([-1, 4, 3, 2])
print(f"my_dot(a, b) = {my_dot(a, b)}")


a = np.array([1, 2, 3, 4])
b = np.array([-1, 4, 3, 2])
c = np.dot(a, b)
print(f"NumPy 1-D np.dot(a, b) = {c}, np.dot(a, b).shape = {c.shape} ") 
c = np.dot(b, a)
print(f"NumPy 1-D np.dot(b, a) = {c}, np.dot(a, b).shape = {c.shape} ")


np.random.seed(1)
a = np.random.rand(10000000)
b = np.random.rand(10000000)

tic = time.time()
c = np.dot(a, b)
toc = time.time()

print(f"np.dot(a, b) =  {c:.4f}")
print(f"Vectorized version duration: {1000*(toc-tic):.4f} ms ")

tic = time.time()
c = my_dot(a, b)
toc = time.time()

print(f"my_dot(a, b) =  {c:.4f}")
print(f"loop version duration: {1000*(toc-tic):.4f} ms ")

del a
del b


X = np.array([[1], [2], [3], [4]])
w = np.array([2])
c = np.dot(X[1], w)

print(f"X[1] has shape {X[1].shape}")
print(f"w has shape {w.shape}")
print(f"c has shape {c.shape}")


a = np.zeros((1, 5))                                       
print(f"a shape = {a.shape}, a = {a}")                     

a = np.zeros((2, 1))                                                                   
print(f"a shape = {a.shape}, a = {a}") 

a = np.random.random_sample((1, 1))  
print(f"a shape = {a.shape}, a = {a}")

a = np.array([[5], [4], [3]])
print(f" a shape = {a.shape}, np.array: a = {a}")
a = np.array([[5],
              [4],
              [3]])
print(f" a shape = {a.shape}, np.array: a = {a}")


a = np.arange(6).reshape(-1, 2)
print(f"a.shape: {a.shape}, \na= {a}")

print(f"\na[2,0].shape:   {a[2, 0].shape}, a[2,0] = {a[2, 0]},     type(a[2,0]) = {type(a[2, 0])} "
      f"Accessing an element returns a scalar\n")

print(f"a[2].shape:   {a[2].shape}, a[2]   = {a[2]}, type(a[2])   = {type(a[2])}")

a = np.arange(20).reshape(-1, 10)
print(f"a = \n{a}")

print("a[0, 2:7:1] = ", a[0, 2:7:1], ",  a[0, 2:7:1].shape =", a[0, 2:7:1].shape, "a 1-D array")

print("a[:, 2:7:1] = \n", a[:, 2:7:1], ",  a[:, 2:7:1].shape =", a[:, 2:7:1].shape, "a 2-D array")

print("a[:,:] = \n", a[:, :], ",  a[:,:].shape =", a[:, :].shape)

print("a[1,:] = ", a[1, :], ",  a[1,:].shape =", a[1, :].shape, "a 1-D array")
print("a[1]   = ", a[1],   ",  a[1].shape   =", a[1].shape, "a 1-D array")
