import numpy as np
#Generate data
# Forward Node
x_F = np.random.rand(4)
y_F = np.random.rand(4)
z_F = np.random.rand(4)
p_F = np.random.rand(4)

#weight
x_y_w = np.random.rand(4,4)
y_z_w = np.random.rand(4,4)
z_p_w = np.random.rand(4,4)
#backward node
x_B = np.random.rand(4)
y_B = np.random.rand(4)
z_B = np.random.rand(4)
p_B = np.random.rand(4)

#TARGET
target = np.array([0.5,0.7,0.3,0.1])
#loss
def SquareErrorLoss(output, target):
    loss = 0
    for i in range(len(output)):
        loss = loss + (output[i] - target[i])**2
    loss = loss
    return loss

# Graident
lr = 0.0000001
for epoch in range(500000):
    # forward0
    y_F = np.matmul(x_F, x_y_w)
    z_F = np.matmul(y_F, y_z_w)
    p_F = np.matmul(z_F, z_p_w)  # 得到输出
    loss_end = SquareErrorLoss(p_F, target)
    # backward
    p_B = 2*p_F  # end grad
    z_B = np.matmul(p_B, z_p_w.T)
    y_B = np.matmul(z_B, y_z_w.T)
    # print(z_F[0])

    # grad
    z_p_w_grad = [np.dot(z_F[0], p_B),
                  np.dot(z_F[1], p_B),
                  np.dot(z_F[2], p_B),
                  np.dot(z_F[3], p_B)]  # 4*4
    y_z_w_grad = [np.dot(y_F[0], z_B),
                  np.dot(y_F[1], z_B),
                  np.dot(y_F[2], z_B),
                  np.dot(y_F[3], z_B)]  # 4*4
    x_y_w_grad = [np.dot(x_F[0], y_B),
                  np.dot(x_F[1], y_B),
                  np.dot(x_F[2], y_B),
                  np.dot(x_F[3], y_B)]  # 4*4
    # update
    x_y_w = x_y_w - lr * np.array(x_y_w_grad)
    y_z_w = y_z_w - lr * np.array(y_z_w_grad)
    z_p_w = z_p_w - lr * np.array(z_p_w_grad)

    if epoch % 5000 == 0:
        print("当前loss值为")
        print(loss_end)
        print(p_F)

#截取输出片段打印
当前loss值为
165.06777280622663
[8.38287281 5.70538503 7.00228248 5.84052431]
当前loss值为
133.11940656598998
[7.56299701 5.15781875 6.33931068 5.28536964]
当前loss值为
109.77775387801975
[6.89894847 4.7147705  5.8028157  4.83622736]
当前loss值为
92.14372363473903
[6.34836332 4.34779557 5.35839007 4.4642465 ]
当前loss值为
78.45934876430728
[5.88318774 4.03806177 4.98325151 4.1503256 ]
当前loss值为
67.60388807433274
[5.48405883 3.77257388 4.66167746 3.88128352]
当前loss值为
58.833090378271265
[5.13715314 3.54205652 4.38244476 3.64771203]
当前loss值为
51.6356735906825
[4.83232077 3.3397007  4.13731374 3.44270456]
... ... ... ... ... ... ... ... ... ... ... ...
当前loss值为
0.34148441938663887
[0.38019982 0.42063884 0.60690939 0.49356868]
当前loss值为
0.33762403308295663
[0.37029135 0.41401346 0.59891746 0.48685882]
当前loss值为
0.3343170568629769
[0.36058041 0.40751086 0.5910724  0.48027121]
当前loss值为
0.3315343883568477
[0.3510625  0.40112821 0.58337076 0.47380298]
当前loss值为
0.3292483287899748
[0.3417333  0.39486274 0.57580922 0.46745137]


