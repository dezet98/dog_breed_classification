import matplotlib.pyplot as plt
import cv2


class Charts:
    @staticmethod
    def plot_history(history, title=None, filename=None):
        plt.rcParams["figure.figsize"] = (16, 5)
        fig, (ax1, ax2) = plt.subplots(1, 2)

        ax1.plot(history['loss'])
        ax1.plot(history['val_loss'])
        ax1.legend(['train', 'test'], loc='upper left')
        ax1.set(xlabel='epoch', ylabel='loss', title='Loss model')

        ax2.plot(history['acc'])
        ax2.plot(history['val_acc'])
        ax2.legend(['train', 'test'], loc='upper left')
        ax2.set(xlabel='epoch', ylabel='accuracy', title='Accuracy model')

        if title is not None:
            fig.suptitle(title, fontsize=16)

        if filename is None:
            plt.show()
        else:
            plt.savefig('results/' + filename)

    @staticmethod
    def plot_dog_prediction(image_path, breed_name, filename=None):
        plt.clf()
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        plt.imshow(img)
        plt.title('org: ' + image_path.split(".")[0] + ' | pred: ' + breed_name)

        if filename is None:
            plt.show()
        else:
            plt.savefig('results/predictions/' + filename)
