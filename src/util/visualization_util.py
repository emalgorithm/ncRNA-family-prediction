import matplotlib.pyplot as plt
import pylab


def plot_loss(train_loss=[], val_loss=[], test_loss=[], file_name='loss.jpg',
              y_label='cross-entropy loss'):
    plt.clf()
    plt.plot(train_loss)
    plt.plot(val_loss)
    plt.plot(test_loss)
    plt.legend(('train loss', 'validation loss', 'test loss'), loc='upper right')
    plt.title('Losses during training of LSTM->LSTM Model')
    plt.xlabel('#epochs')
    plt.ylabel(y_label)
    # plt.show()
    pylab.savefig(file_name)
