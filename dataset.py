import numpy as np
import matplotlib.pyplot as plt
import random

class TutorialDatasetManager:
    def __init__(self, dataset_size=100, width=32, height=32, noise_u=0.0, noise_sd=0.2, target_r=2.8, target_value=1.5):
        self.dataset_size = dataset_size
        self.width = width
        self.height = height
        self.noise_u = noise_u
        self.noise_sd = noise_sd
        self.target_r = target_r
        self.target_value = target_value
        self.generate_dataset()

    def normal_rnd(self):
        return np.random.normal(self.noise_u, self.noise_sd)

    def generate_data(self):
        x = np.array([[self.normal_rnd() for x in range(self.width)] for y in range(self.height)], np.float32)

        px = np.random.rand() * (self.width - 2.0 * self.target_r) + self.target_r
        py = np.random.rand() * (self.height - 2.0 * self.target_r) + self.target_r

        for y_i in range(self.height):
            for x_i in range(self.width):
                if (px - x_i) ** 2 + (py - y_i) ** 2 < self.target_r ** 2:
                    x[y_i][x_i] = self.target_value
        return px, py, np.reshape(x, [self.width * self.height])

    def generate_dataset(self):
        data_shape = [self.dataset_size, self.width * self.height]

        self.dataset_x = np.zeros(shape=data_shape)
        self.dataset_y = np.zeros(shape=[self.dataset_size, 2])
        for i in range(self.dataset_size):
            px, py, x = self.generate_data()
            self.dataset_x[i] = x
            self.dataset_y[i] = np.array([px, py])
    
    def next_batch(self, size):
        idx = [np.random.randint(low=0, high=self.dataset_size) for _ in range(size)]
        return self.dataset_x[idx], self.dataset_y[idx]

if __name__ == "__main__":
    dataset_manager = TutorialDatasetManager()
    x, y = dataset_manager.next_batch(10)

    plt_nrow = 3
    plt_ncol = 3
    fig = plt.figure()
    for i in range(plt_nrow * plt_ncol):
        sub = fig.add_subplot(plt_nrow, plt_ncol, i + 1)
        sub.set_title(y[i])
        plt.imshow(np.reshape(x[i], (32, 32)), cmap="gray")
        plt.plot(y[i][0], y[i][1], "+r")
    plt.show()
