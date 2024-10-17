import numpy as np
import Net1
import read_data


net = Net1.Net([28*28, 512, 10])
net.load('weights')


def valid():
    test_loader = read_data.from_csv('../mnist-in-csv/versions/2/mnist_test.csv', 64)
    tot_cnt = 0
    right_cnt = 0
    for data, labels in test_loader:
        y_pred = net.forward(data)
        y_pred = np.argmax(y_pred, axis=1)
        right_cnt += np.sum(y_pred == labels)
        tot_cnt += len(y_pred)
    acc = right_cnt / tot_cnt
    print(f"accuracy: {acc}")


if __name__ == '__main__':
    valid()
