from matplotlib import pyplot as plt


class RightTriangle:
    def __init__(self, a, b):
        self.a = a
        self.b = b

        self.c = (a ** 2 + b ** 2) ** 0.5

    def plot_triangle(self):
        plt.plot()


if __name__ == "__main__":
    triangle1 = RightTriangle(1, 1)
    print(triangle1.c)
    triangle2 = RightTriangle(3, 4)
    print(triangle2.c)
