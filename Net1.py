import numpy as np


class Net:
    def __init__(self, shape: list, lr=0.007):
        self.shape = shape
        self.lr = lr
        self.w = [None]
        self.b = [None]
        self.a = []
        for i in range(1, len(shape)):
            self.w.append(np.random.randn(shape[i - 1], shape[i]) * 0.01)
            self.b.append(np.random.randn(shape[i]) * 0.01)

    def relu(self, x: np.ndarray):
        return np.maximum(x, 0)

    def softmax(self, x: np.ndarray):
        x1 = np.exp(x - np.max(x, axis=1, keepdims=True))
        sum1 = np.sum(x1, axis=1, keepdims=True)
        return x1 / sum1

    def forward(self, x: np.ndarray):  # batched version
        self.a = [x]
        for i in range(1, len(self.shape)):
            z = np.dot(self.a[-1], self.w[i]) + self.b[i]
            if i == len(self.shape) - 1:
                self.a.append(self.softmax(z))
                break
            a = self.relu(z)
            self.a.append(a)
        return self.a[-1]

    def backward(self, y_true: np.ndarray):
        grad_z = self.a[-1] - y_true  # 第i层激活函数前
        for i in range(len(self.shape) - 1, 0, -1):
            grad_w = np.dot(self.a[i - 1].T, grad_z) / len(grad_z)
            grad_b = np.sum(grad_z, axis=0) / len(grad_z)
            # 更新参数
            self.w[i] -= grad_w * self.lr
            self.b[i] -= grad_b * self.lr
            # 更新grad_a
            temp = np.dot(grad_z, self.w[i].T)
            grad_z = temp * (self.a[i - 1] > 0)

    def dump(self, pth):
        s = ''
        for i in range(1, len(self.w)):
            for j in range(self.w[i].shape[0]):
                for k in range(self.w[i].shape[1]):
                    s += str(self.w[i][j][k])
                    s += ' '
        for i in range(1, len(self.b)):
            for j in range(self.b[i].shape[0]):
                s += str(self.b[i][j])
                s += ' '
        with open(pth, 'w') as f:
            f.write(s)

    def load(self, pth):
        with open(pth, 'r') as f:
            s = f.read()
        s = s.split(' ')
        tail = 0
        for i in range(1, len(self.w)):
            for j in range(self.w[i].shape[0]):
                for k in range(self.w[i].shape[1]):
                    self.w[i][j][k] = float(s[tail])
                    tail += 1
        for i in range(1, len(self.b)):
            for j in range(self.b[i].shape[0]):
                self.b[i][j] = float(s[tail])
                tail += 1
