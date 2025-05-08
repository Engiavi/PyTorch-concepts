import torch # firstly import the pytorch library

# to work with torch we have to create two tensor x and y

x = torch.tensor(3.0, requires_grad=True) # to create a tensor as derivative we have to explicitly mentioned requires_grad = True
# after running above command then it internally creates a computational graph
y = x ** 2

# print("x:",x,"\n","y:",y) # print the value of x and y

# to calculate the derivative we have to call backward() function
y.backward() # this will calculate the derivative of y with respect to x
# print("x.grad:",x.grad) # this will print the derivative of y with respect to x

#  now we are creating three tensor x, y and z
x = torch.tensor(3.0, requires_grad=True)
y = x ** 2
z = torch.sin(y) # this will create a new tensor z which is the sine of y

# print("x:",x,"\n","y:",y,"\n","z:",z) # print the value of x, y and z

z.backward()

# print("derivative: ", x.grad)

# code for complex differentiation using manual calculation
m = torch.tensor(6.7) # input feature
n = torch.tensor(0.0) # true label(binary)

w = torch.tensor(1.0) # weight
b = torch.tensor(0.0) # bias

# Binary Cross-Entropy Loss for scalar
def binary_cross_entropy_loss(prediction, target):
    epsilon = 1e-8  # To prevent log(0)
    prediction = torch.clamp(prediction, epsilon, 1 - epsilon)
    return -(target * torch.log(prediction) + (1 - target) * torch.log(1 - prediction))

# Forward pass
z = w * m + b  # Weighted sum (linear part)
y_pred = torch.sigmoid(z)  # Predicted probability

# Compute binary cross-entropy loss
loss = binary_cross_entropy_loss(y_pred, n)

# print("loss:", loss)  # Print the loss value

# Derivatives:
# 1. dL/d(y_pred): Loss with respect to the prediction (y_pred)
dloss_dy_pred = (y_pred - n)/(y_pred*(1-y_pred))

# 2. dy_pred/dz: Prediction (y_pred) with respect to z (sigmoid derivative)
dy_pred_dz = y_pred * (1 - y_pred)

# 3. dz/dw and dz/db: z with respect to w and b
dz_dw = m  # dz/dw = x
dz_db = 1  # dz/db = 1 (bias contributes directly to z)

dL_dw = dloss_dy_pred * dy_pred_dz * dz_dw
dL_db = dloss_dy_pred * dy_pred_dz * dz_db

print(f"Manual Gradient of loss w.r.t weight (dw): {dL_dw}")
print(f"Manual Gradient of loss w.r.t bias (db): {dL_db}")

# code for complex differentiation using autograd calculation

x = torch.tensor(6.7)
y = torch.tensor(0.0)
w = torch.tensor(1.0, requires_grad=True)
b = torch.tensor(0.0, requires_grad=True)
z = w*x + b
y_pred = torch.sigmoid(z)
loss = binary_cross_entropy_loss(y_pred, y)
loss.backward()
print(w.grad)
print(b.grad)