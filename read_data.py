import numpy as np


def from_csv(path, batch_size):
    with open(path, 'r') as f:
        s = f.read()
    data = []
    labels = []
    for line in s.split('\n')[1::]:
        if len(line) == 0:
            break
        flag = False
        temp = []
        for num in line.split(','):
            if not flag:
                labels.append(int(num))
                flag = True
            else:
                temp.append(float(num))
        data.append(temp)
    dataset = []
    for i in range(0, len(data), batch_size):
        temp = [np.array(data[i: i + batch_size: 1]), np.array(labels[i: i + batch_size: 1])]
        dataset.append(temp)
    return dataset
