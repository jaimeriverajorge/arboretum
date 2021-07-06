# tensor indexing
import torch

batch_size = 10
features = 25
x = torch.rand((batch_size, features))
# get all the features (columns) of the first batch (row)
print(x[0].shape)
# get the first feature of all the batches
print(x[:, 0].shape)

# get the first 10 features of the third batch
print(x[2, 0:10].shape)

# assign the first index (feature of first batch size) to 100
x[0, 0] = 100

# fancy indexing
x = torch.arange(10)
indices = [2, 5, 8]
# get the values in the indices 2,5,8
print(x[indices])

x = torch.rand((3, 5))
rows = torch.tensor([1, 0])
cols = torch.tensor([4, 0])
# get the values of 1,4 and 0,0
print(x[rows, cols].shape)

# more advanced indexing
x = torch.arange(10)
# get the elements less than 2 or greater than 8
print(x[(x < 2) | (x > 8)])
# print the elements where remainder of 2 is 0 (even elements)
print(x[x.remainder(2) == 0])

# useful operations
# return x for the indices where x >5, return
# x * 2 where x < 5
print(torch.where(x > 5, x, x*2))
# print only the unique values within the tensors
print(torch.tensor([0, 0, 1, 2, 2, 3, 4]).unique())
# print the number of dimensions that x has
print(x.ndimension())
# print the number of elements in x
print(x.numel())
