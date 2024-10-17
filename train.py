import numpy as np
import Net1
import read_data


batch_size = 64
class_num = 10
epoch = 40
train_loader = read_data.from_csv('../mnist-in-csv/versions/2/mnist_train.csv', batch_size)
net = Net1.Net([28*28, 512, 10])


def one_hot(x: np.ndarray, tot):
    ret = np.zeros([len(x), tot])
    for i in range(len(x)):
        ret[i][int(x[i])] = 1.
    return ret


def mse_loss(y_pred: np.ndarray, y_true: np.ndarray):
    return np.sum(np.square(y_pred - y_true)) / np.sum(np.ones(y_pred.shape))


def init_data():
    for i in range(len(train_loader)):
        train_loader[i][0] /= 255


def train(idx):
    loss = 0.
    cnt = 0
    for data, labels in train_loader:
        y_pred = net.forward(data)
        y_true = one_hot(labels, class_num)
        loss += mse_loss(y_pred, y_true)
        cnt += 1
        net.backward(y_true)
    loss /= cnt
    print(f"{idx}th loss: {loss}")


def valid():
    test_loader = read_data.from_csv('../mnist-in-csv/versions/2/mnist_test.csv', batch_size)
    tot_cnt = 0
    right_cnt = 0
    for data, labels in test_loader:
        y_pred = net.forward(data)
        y_pred = np.argmax(y_pred, axis=1)
        right_cnt += np.sum(y_pred == labels)
        tot_cnt += len(y_pred)
    acc = right_cnt / tot_cnt
    print(f"accuracy: {acc}")


if __name__ == "__main__":
    init_data()
    for i in range(epoch):
        train(i)
    net.dump('weights')
    valid()
