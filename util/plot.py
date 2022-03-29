from cProfile import label
import matplotlib.pyplot as plt

def plot_loss(loss_history):
    plt.plot(loss_history, '-r', label='loss')
    plt.show()


if __name__ == '__main__':
    loss_history = [1.2, 1, 0.4, 0.3, 0.1, 0.001, 0.0005]
    plot_loss(loss_history)