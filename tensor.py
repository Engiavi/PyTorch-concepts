import torch

# if torch.cuda.is_available():
#     print("GPU is available!")
#     print(f"Using GPU: {torch.cuda.get_device_name(0)}")
# else:
#     print("GPU not available. Using CPU.")
    
# Create a tensor

# using empty
x = torch.empty(5, 3)
# print(x)

# check the type of variable
# print(type(x))

# using zeros
torch.zeros(2,3)

# using ones
torch.ones(2,3)

# use of seed
torch.rand(2,3) # random numbers between 0 and 1

# manual_seed, use when you want to reproduce the same random numbers
torch.manual_seed(100)
torch.rand(2,3)

# using tensor, you can create a tensor of your own values
torch.tensor([[1,2,3],[4,5,6]])

# other ways

# arange
print("using arange ->", torch.arange(0,10,2))

# using linspace
print("using linspace ->", torch.linspace(0,10,10))

# using eye
print("using eye ->", torch.eye(5))

# using full
print("using full ->", torch.full((3, 3), 5))

# Tensor Shapes
a = torch.tensor([[1,2,3],[4,5,6]])
# these likes methods are used to get the shape of the tensor and create tensor of same shape and provided methods like empty creates random values, zeros creates tensor of zeros, ones creates tensor of ones and rand creates random values between 0 and 1
# 
torch.empty_like(x)

torch.zeros_like(x)

torch.ones_like(x)

torch.rand_like(x, dtype=torch.float32)
