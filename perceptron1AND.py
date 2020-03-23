import numpy as np
import matplotlib.pyplot as plt
import time

#bemeneti adat
input_array = np.array([[0, 0, 0],
                        [0, 1, 0],
                        [1, 0, 0],
                        [1, 1, 1]])

#AND kapu logikai kimenete
output_array = [0, 0, 0, 1]


#inicializalja a kirajzolando figurat
def init_figure(data):
    plt.ion()
    figure = plt.figure()
    figure.suptitle('Perceptron for AND gate')
    plt.xlabel('xlabel')
    plt.ylabel('ylabel')
    plt.xlim((-0.5, 1.5))
    plt.ylim((-0.5, 1.5))
    plt.grid(True)
    plt.scatter(data[:, 1], data[:, 2])
    return figure


def hard_lim(val):
    if val < 0:
        return 0
    else:
        return 1


def perceptron_learning(data, expected_output):
    N, n = data.shape
    lr = .1
    w = np.random.randn(n, 1)
    E = 1

    figure = init_figure(data)
    x = np.linspace(-5, 5, 50)

    while E != 0:
        E = 0

        for i in range(N):
            yi = hard_lim(np.dot(data[i], w))
            ei = expected_output[i] - yi
            w += lr * ei * data[i].reshape(n, 1)
            E += ei ** 2

        a = [0, -w[0] / w[2]]
        c = [-w[0] / w[1], 0]
        m = (a[1] - a[0]) / (c[1] - c[0])
        line, = plt.plot(x, x * m + a[1])
        line.set_ydata(x * m + a[1])
        figure.canvas.draw()
        time.sleep(1)
        figure.canvas.flush_events()


perceptron_learning(input_array, output_array)