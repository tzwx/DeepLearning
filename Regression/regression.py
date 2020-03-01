#写一个深度学习的回归案例
#y = w1*x^2 + w2*x + b ,计算出参数 w1 w2 b
import numpy as np

#生成数据 
x_data = np.random.rand(10)
w1_truth = 1.8
w2_truth = 2.4
b_truth = 5.6
y_data = np.random.rand(10)
for i in range(10):
    y_data[i] = w1_truth*x_data[i]*x_data[i] + w2_truth*x_data[i] + b_truth
print(y_data.shape)
print(x_data.shape)

# 设置参数
w1 = 6
w2 = 4
b = 8
lr = 1
steps = 100000
lr_w1 = 0
lr_w2 = 0
lr_b = 0
for epoch in range(steps):
    w1_grad = 0
    w2_grad = 0
    b_grad = 0
    for i in range(10):
        w1_grad = w1_grad - 2*(y_data[i] - b-w1*x_data[i]*x_data[i]-w2*x_data[i])* (x_data[i]**2)
        w2_grad = w2_grad - 2*(y_data[i] - b-w1*x_data[i]*x_data[i]-w2*x_data[i])* x_data[i]
        b_grad = b_grad - 2*(y_data[i] - b-w1*x_data[i]*x_data[i]-w2*x_data[i])* 1.0
    lr_w1 = lr_w1 + w1_grad**2
    lr_w2 = lr_w2 + w2_grad**2
    lr_b = lr_b + b_grad**2
    # update
    w1 = w1 - lr/np.sqrt(lr_w1) * w1_grad
    w2 = w2 - lr/np.sqrt(lr_w2) * w2_grad
    b  = b  - lr/np.sqrt(lr_b) * b_grad
print("w1= %f,w2 = %f, b= %f" % (w1,w2,b))
#以下是代码的输出结果，可以自己测试一下，完美找到了真实值
# w1= 1.800000,w2 = 2.400000, b= 5.600000


