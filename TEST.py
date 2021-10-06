import numpy as np
import pandas as pd


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


class Route:
    def __init__(self, address_list):
        self.address_list: list[City] = address_list
        self.calculated = False  # показывает, если матрица уже посчитана
        self.current_matrix = None  # хранение посчитанной матрицы
        self.num_of_adr = len(self.address_list)
        self._H = 0  # локальная нижняя граница
        self._route = list()

        self._num_of_rows = [i for i in range(self.num_of_adr)]
        self._num_of_columns = [i for i in range(self.num_of_adr)]

    @property
    def matrix(self) -> np.ndarray:
        """
        Геттер.
        Матрица кратчайших расстояний.
        """
        if self.calculated:
            print('матрица уже посчитана!')
            return self.current_matrix

        number: int = len(self.address_list)
        matrix: np.array = np.zeros([number, number])  # нулевая матрица
        for i in range(number):
            for j in range(number):
                if i == j:
                    matrix[i, j] = float('inf')  # Заполнение главной диагонали матрицы
                else:
                    matrix[i, j] = round(self.address_list[i].distance(self.address_list[j]), 3)

        self.calculated = True
        self.current_matrix = matrix
        return matrix

    @property
    def route(self):
        return self._route

    def print_matrix(self):
        matrix = self.current_matrix
        rows_names = self._num_of_rows
        column_names = self._num_of_columns

        df = pd.DataFrame(matrix, index=rows_names, columns=column_names)
        print("Матрица: ")
        print(df)

    def reduction(self):
        """
        Нахождение минимума по строкам.
        Редукция строк.
        Нахождение минимума по столбцам.
        Редукция столбцов.
        Обновляем локальную нижнюю границу.
        """

        # Вычитаем минимальный элемент в строках
        for i_row in range(self.num_of_adr):
            row_min = min(i for i in self.current_matrix[i_row])  # min эдемент строки
            self._H += row_min
            for i_elem in range(self.num_of_adr):
                self.current_matrix[i_row][i_elem] -= row_min

        # Вычитаем минимальный элемент в столбцах
        for i_column in range(self.num_of_adr):
            column_min = min(row[i_column] for row in self.current_matrix)
            self._H += column_min
            for i_elem in range(self.num_of_adr):
                self.current_matrix[i_elem][i_column] -= column_min

    # Функция нахождения минимального элемента, не включая текущего
    def find_min_elem(self, lst, cur_index):
        return min(x for index, x in enumerate(lst) if index != cur_index)

    def matrix_reduce(self, row_i: int, column_i: int) -> None:
        print(f'Удаляем:\nстроку {row_i}'
              f'\nстолбец {column_i}')
        print(self._num_of_rows[column_i])
        print(self._num_of_columns[row_i])

        del self._num_of_rows[column_i]
        del self._num_of_columns[row_i]
        self.num_of_adr -= 1
        self.current_matrix = np.delete(self.current_matrix, [row_i], 0)
        self.current_matrix = np.delete(self.current_matrix, [column_i], 1)

    def evaluation_of_zero_cells(self):
        """
        Оцениваем нулевые клетки и ищем в матрице нулевую клетку с максимальной оценкой.
        """
        # максимальная оценка нулевой клетки
        zero_value = 0
        # её координаты в матрице
        x = 0
        y = 0

        tmp = 0

        for i_row in range(self.num_of_adr):
            for i_column in range(self.num_of_adr):
                if self.current_matrix[i_row][i_column] == 0:
                    tmp = self.find_min_elem(self.current_matrix[i_row], i_column) + self.find_min_elem(
                        (row[i_column] for row in self.current_matrix), i_row)

                    if tmp >= zero_value:
                        zero_value = tmp
                        x = i_row
                        y = i_column

        # Находим нужный нам путь, записываем его в res и удаляем все ненужное
        self._route.append(self._num_of_rows[y])
        self._route.append(self._num_of_columns[x])

        # в клетку обратного пути устанавливаем inf,
        # так как мы уже не будем возвращаться обратно из y в x.
        self.current_matrix[x][y] = float('inf')


        print(self._H)
        return y, x

    def run(self):
        a = self.matrix
        while True:
            self.print_matrix()
            self.reduction()
            self.print_matrix()
            print(self._route)
            result = self.evaluation_of_zero_cells()
            r.matrix_reduce(result[0], result[1])
            if len(self.current_matrix) == 1: break

        print(self._H)
        print(self._route)


if __name__ == '__main__':
    c0 = City(0, 2)
    c1 = City(2, 5)
    c2 = City(5, 2)
    c3 = City(6, 6)
    c4 = City(8, 3)

    r = Route([c0, c1, c2, c3, c4])

    r.run()


