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
    def __init__(self, address_list=None, current_matrix=None):
        if address_list:
            self.address_list: list[City] = address_list
            self.calculated = False  # показывает, если матрица уже посчитана
            self.num_of_adr = len(self.address_list)  # кол-во адресов
            self.current_matrix = current_matrix  # хранение посчитанной матрицы

        elif current_matrix:
            self.current_matrix = current_matrix  # хранение посчитанной матрицы
            self.calculated = True  # показывает, если матрица уже посчитана
            self.num_of_adr = len(self.current_matrix)  # кол-во адресов

        self._H = 0  # локальная нижняя граница
        self._route = list()
        self._distance = 0  # Общая дистанция пути

        self.row_indexes = [i for i in range(self.num_of_adr)]
        self.column_indexes = [i for i in range(self.num_of_adr)]

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
        rows_names = self.row_indexes
        column_names = self.column_indexes

        df = pd.DataFrame(matrix, index=rows_names, columns=column_names)
        print("Матрица: ")
        print(df)

    def row_reduction(self):
        """
        Нахождение минимума по строкам -> Редукция строк.
        Обновляем локальную нижнюю границу.
        """

        # Вычитаем минимальный элемент в строках
        for i_row in range(self.num_of_adr):
            row_min = min(i for i in self.current_matrix[i_row])  # min эдемент строки
            self._H += row_min
            for i_elem in range(self.num_of_adr):
                self.current_matrix[i_row][i_elem] -= row_min

            print(f'{i_row} Row min: ', row_min)

        self._distance += self._H

    def column_reduction(self):
        """
        Нахождение минимума по столбцам -> Редукция столбцов.
        Обновляем локальную нижнюю границу.
        """

        # Вычитаем минимальный элемент в столбцах
        for i_column in range(self.num_of_adr):
            column_min = min(row[i_column] for row in self.current_matrix)
            self._H += column_min
            for i_elem in range(self.num_of_adr):
                self.current_matrix[i_elem][i_column] -= column_min
            print(f'{i_column} column min: ', column_min)
        print('-' * 20)
        print(f'H_0 = {self._H}')
        print('-' * 20)

    #
    def find_min_elem(self, lst, cur_index):
        """ Функция нахождения минимального элемента, не включая текущего """

        return min(x for index, x in enumerate(lst) if index != cur_index)

    def evaluation_of_zero_cells(self):
        """
        Оцениваем нулевые клетки и ищем в матрице нулевую клетку с максимальной оценкой.
        На выходе индекс максимальной оценки нулевой клетки
        """
        # максимальная оценка нулевой клетки
        zero_value = 0
        # её координаты в матрице
        row_index = 0
        column_index = 0

        tmp = 0

        for i_row in range(self.num_of_adr):
            for i_column in range(self.num_of_adr):
                if self.current_matrix[i_row][i_column] == 0:
                    min_in_cur_row = self.find_min_elem(self.current_matrix[i_row], i_column)
                    min_in_cur_column = self.find_min_elem((row[i_column] for row in self.current_matrix), i_row)
                    tmp = min_in_cur_row + min_in_cur_column
                    if tmp > zero_value:
                        zero_value = tmp
                        row_index = i_row
                        column_index = i_column

        print('Row index:', row_index, 'Column index:', column_index, "value: ", zero_value)

        return row_index, column_index, zero_value

    def supplement_the_route(self, row_index, column_index, zero_value):
        self._route.append([self.row_indexes[row_index], self.row_indexes[column_index]])
        self._distance += zero_value
        print(f'New route = {self._route}')
        print(f'h_0 = {self._H}')
        print(f'distance: {self._distance}')
        print(column_index)

    def matrix_reduce(self, row_i: int, column_i: int) -> None:
        print(f'Удаляем:\nстроку {self.row_indexes[row_i]}'
              f'\nстолбец {self.column_indexes[column_i]}')

        self.current_matrix[column_i][row_i] = float('inf')
        del self.row_indexes[row_i]
        del self.column_indexes[column_i]
        self.num_of_adr -= 1
        self.current_matrix = np.delete(self.current_matrix, [row_i], 0)
        self.current_matrix = np.delete(self.current_matrix, [column_i], 1)

    # def run(self):
    #     a = self.matrix
    #     while True:
    #         self.print_matrix()
    #         self.reduction()
    #         self.print_matrix()
    #         print(self._route)
    #         result = self.evaluation_of_zero_cells()
    #         r.matrix_reduce(result[0], result[1])
    #         if len(self.current_matrix) == 1: break
    #
    #     print(self._H)
    #     print(self._route)


if __name__ == '__main__':
    pass