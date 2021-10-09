import numpy as np


class City:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def distance(self, city):
        xDis = abs(self.x - city.x)
        yDis = abs(self.y - city.y)
        distance = np.sqrt((xDis ** 2) + (yDis ** 2))
        return distance

    def __repr__(self):
        return "(" + str(self.x) + "," + str(self.y) + ")"


class Matrix:
    def __init__(self, address_list: list[City]) -> None:
        self.address_list: list[City] = address_list
        self.address_list = address_list
        self.current_matrix = None

    @property
    def matrix(self) -> np.ndarray:
        number: int = len(self.address_list)
        matrix: np.array = np.zeros([number, number])  # нулевая матрица
        for i in range(number):
            for j in range(number):
                if i == j:
                    matrix[i, j] = float('inf')  # Заполнение главной диагонали матрицы
                else:
                    matrix[i, j] = round(self.address_list[i].distance(self.address_list[j]), 3)

        self.current_matrix = matrix
        return matrix
