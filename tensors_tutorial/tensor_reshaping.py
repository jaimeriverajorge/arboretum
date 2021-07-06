# Tensor reshaping
import torch

x = torch.arange(9)
# rearrange x to be a 3x3 matrix
# view and reshape do almost the same thing
# view acts on contigous tensors, stored contiguosly in memory
x_3x3 = x.view(3, 3)
print(x_3x3)
x_33 = x.reshape(3, 3)
# make the transpose of x_3x3 and set it to x
# transpose is a special case of permute, only usable with a matrix
y = x_3x3.t()
# this gives an error because y spans across 2 dimensions
#y_9x9 = y.view(9)
y_9x9 = y.reshape(9)
print(y)
print(y_9x9)

x1 = torch.rand((2, 5))
x2 = torch.rand((2, 5))
# concatenate the matrices along the first dimension
print(torch.cat((x1, x2), dim=0).shape)
# concatenate the matrices along the second dimension
print(torch.cat((x1, x2), dim=1).shape)

# unroll the elements in x1 to get 10 elements across
z = x1.view(-1)
print(z.shape)

batch = 64
x = torch.rand((batch, 2, 5))
# this keeps the batch dimension and unrolls the rest
z = x.view(batch, -1)
print(z.shape)

# switch two dimensions
z = x.permute(0, 2, 1)
print(z.shape)

x = torch.arange(10)  # [10]
# unsqueeze adds a dimension at the specified index
print(x.unsqueeze(0).shape)

x = torch.arange(10).unsqueeze(0).unsqueeze(1)  # 1x1x10
# remove one of the indexes
z = x.squeeze(1)
print(z.shape)
