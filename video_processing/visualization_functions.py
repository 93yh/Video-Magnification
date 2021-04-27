import matplotlib.pyplot as plt
import numpy as np


def plot_components_or_sources(rows, columns, t, freq, visualize, order):
    fig, axs = plt.subplots(rows, 3)
    for row in range(rows):
        for column in range(columns):
            axs[row][0].plot(t, visualize[:, order[row]], color="#069AF3")

            aux = (np.abs(np.fft.fft(visualize[:, order[row]]))) ** 2
            axs[row][1].plot(freq[2:300], aux[2:300], color="#069AF3")

            aux = (np.angle(np.fft.fft(visualize[:, order[row]]))) ** 2
            axs[row][2].plot(freq[2:300], aux[2:300], color="#069AF3")
    plt.show()


def plot_mode_shapes_and_modal_coordinates(info, columns, t, do_unscramble):
    fig2, axs2 = plt.subplots(2, columns)
    for column in range(columns):
        axs2[0][column].plot(t, info.modal_coordinates[:, column], color="#069AF3")
        if not do_unscramble:
            axs2[1][column].imshow(info.mode_shapes[:, column].reshape(info.video.frames_shape, order="F"),
                                   'gray', aspect='auto')
        else:
            mode_shape = info.mode_shapes[info.encryption_key, column]
            axs2[1][column].imshow(mode_shape.reshape(info.video.frames_shape, order="F"), 'gray', aspect='auto')
    plt.show()
