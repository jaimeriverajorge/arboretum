# tensor math and comparison operations
import torch

x = torch.tensor([1, 2, 3])
y = torch.tensor([9, 8, 7])

# Addition
z1 = torch.empty(3)
torch.add(x, y, out=z1)
z2 = torch.add(x, y)
z = x + y

# Subtraction
z = x - y

# Division
z = torch.true_divide(x, y)

# in-place operations, can be more computationnally efficient
# denoted by the underscore after the function call
t = torch.zeros(3)
t.add_(x)
t += x  # t = t + x will not be in place, creates a copy first

# Exponentiation, element wise
z = x.pow(2)
z = x ** 2

# simple comparison (element wise)
z = x > 0
z = x < 0

# Matrix multiplication
x1 = torch.rand((2, 5))
x2 = torch.rand((5, 3))
x3 = torch.mm(x1, x2)
x3 = x1.mm(x2)

# Matrix exponentiation
matrix_exp = torch.rand(5, 5)
matrix_exp.matrix_power(3)

# Element wise multiplication in matrix
z = x * y

# dot product
z = torch.dot(x, y)

# Batch Matrix multiplication
batch = 6
n = 1
m = 2
p = 3

tensor1 = torch.rand((batch, n, m))
tensor2 = torch.rand((batch, m, p))
out_bmm = torch.bmm(tensor1, tensor2)  # (batch, n, p)

# Example of broadcasting
x1 = torch.rand((5, 5))
x2 = torch.rand((1, 5))
# x2 is automatically expanded to perform the required operation
z = x1 - x2

# Other useful tensor operations
# return the sum of the values in the tensor across a specified dimension
sum_x = torch.sum(x, dim=0)
# returns the value and index of the maximum value in a tensor,
# across a specified dimension
values, indices = torch.max(x, dim=0)
values, indices = torch.min(x, dim=0)
# returns the absolute value (element-wise) of x
abs_x = torch.abs(x)
# returns only the index where the maximum value of a tensor is
arg_max = torch.argmax(x, dim=0)
arg_min = torch.argmin(x, dim=0)
# find the mean, x must be a float first
mean_x = torch.mean(x.float(), dim=0)
# element-wise compare two vectors / matriices
z = torch.eq(x, y)
# sort using torch.sort, either ascending or descending order
# returns both the values and the indices of the values
sorted_y = torch.sort(y, dim=0, descending=False)
# checks all elements of x that are less than 0, and sets them to 0
# or values that are greater than 10, and sets them to 10
z = torch.clamp(x, min=0, max=10)
r = torch.clamp(x, min=0)  # this is essentially the ReLu function

x = torch.tensor([1, 0, 1, 1, 1], dtype=torch.bool)
# checks if ANY value within x is true
z = torch.any(x)
# checks if ALL values within x are true
z = torch.all(x)
