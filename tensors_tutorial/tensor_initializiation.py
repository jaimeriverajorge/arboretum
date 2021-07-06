# a tutorial on using tensors in pytorch

import numpy as np
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
my_tensor = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32,
                         device=device)
# device wont show if on cpu
# print(my_tensor)
# print the type of the values inside the tensor
# print(my_tensor.dtype)
# print the device that the tensor is on
# print(my_tensor.device)
# print the shape of the tensor
# print(my_tensor.shape)

# other common initialization methods
# empty tensor of size 3 x 3
x = torch.empty(size=(3, 3))
# tensor filled with zero values
x = torch.zeros((3, 3))
# tensor filled with random values between 0 and 1
x = torch.rand((3, 3))
# tensor flled with 1 values
x = torch.ones((3, 3))
# an identity matrix
x = torch.eye(5, 5)
# tensor starting at value 0, ending at 5, with step size at 1
x = torch.arange(start=0, end=5, step=1)
# tensor staring at value 0.1, ending at 1, with 10 steps to get there
x = torch.linspace(start=0.1, end=1, steps=10)
# tensor with empty values using normal distribution
x = torch.empty(size=(1, 5)).normal_(mean=0, std=1)
# tensor with diagonal values of 1, same as identity matrix
x = torch.diag(torch.ones(3))

# how to initialize and convert tensors to other types (int, float, double)
# start is 0 by default, step is 1 by default, only end is 4
tensor = torch.arange(4)
# prints the tensor converted to boolean values (0 = false, >0 = true)
print(tensor.bool())
# prints the tensor converted to float64 values
tensor = tensor.long()
# conveerts tensor to float32 values (used often)
tensor = tensor.float()

# array to tensor conversion and vice-versa
np_array = np.zeros((5, 5))
tensor = torch.from_numpy(np_array)
np_array_back = tensor.numpy()
