import torch # firstly import the pytorch library

# to work with torch we have to create two tensor x and y

x = torch.tensor(3.0, requires_grad=True) # to create a tensor as derivative we have to explicitly mentioned requires_grad = True
# after running above command then it internally creates a computational graph
y = x ** 2

# print("x:",x,"\n","y:",y) # print the value of x and y

# to calculate the derivative we have to call backward() function
y.backward() # this will calculate the derivative of y with respect to x
print("x.grad:",x.grad) # this will print the derivative of y with respect to x
