import numpy as np
import pandas as pd


class Node:
    ID = 0
    NODES = []

    def __init__(self, matrix, ib=None):
        self.value = 0
        self.id = Node.ID
        # дочерние узлы
        self.next = None

        # родительский узел
        self.root = None

        self.route = 0

        # номер первого города
        self.ib = 0

        # матрица кратчайших расстояний
        self.matrix = matrix

        # кол-во вершин
        self.num = len(matrix)

        self.row_indexes = [i for i in range(self.num)]
        self.column_indexes = [i for i in range(self.num)]

        Node.ID += 1
        Node.NODES.append(self)

    @classmethod
    def find_min_elem(cls, lst, cur_index):
        """
        Функция нахождения минимального элемента,
        не включая текущего
        """

        return min(x for index, x in enumerate(lst) if index != cur_index)

    def check_child(self):
        if self.next:
            print(self.next.id)
        else:
            print('0')

    def check_parameters(self):

        print('=====' * 25)
        if self.root is None:
            root_id = 'я корень!'
        else:
            root_id = self.root.id
        if self.next.id is None:
            child = 0
        else:
            child = self.next.id

        print(f'Я {self.id}')
        print('____' * 25)
        print(f'Мой отец: {root_id}, мой сын: {child}')

        self.print_matrix()
        print(f'== Value: {self.value} ==')
        print(f'== Way: {self.route} ==')

    def print_matrix(self):
        matrix = self.matrix
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
        for i_row in range(self.num):
            row_min = min(i for i in self.matrix[i_row])
            if row_min == float('inf'):
                continue
            self.value += row_min
            for i_elem in range(self.num):
                self.matrix[i_row][i_elem] -= row_min

    def column_reduction(self):
        """
        Нахождение минимума по столбцам -> Редукция столбцов.
        Обновляем локальную нижнюю границу.
        """
        # Вычитаем минимальный элемент в столбцах
        for i_column in range(self.num):
            column_min = min(row[i_column] for row in self.matrix)
            if column_min == float('inf'):
                continue
            self.value += column_min
            for i_elem in range(self.num):
                self.matrix[i_elem][i_column] -= column_min

    def new_node(self, node, row_indexes, column_indexes):
        self.root = node
        self.row_indexes = row_indexes
        self.column_indexes = column_indexes

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

        for i_row in range(self.num):
            for i_column in range(self.num):
                if self.matrix[i_row][i_column] == 0:
                    min_in_cur_row = self.find_min_elem(self.matrix[i_row], i_column)
                    min_in_cur_column = self.find_min_elem((row[i_column] for row in self.matrix), i_row)

                    tmp = min_in_cur_row + min_in_cur_column
                    if tmp > zero_value:
                        zero_value = tmp
                        row_index = i_row
                        column_index = i_column

        return row_index, column_index

    def matrix_reduce(self, row_i: int, column_i: int) -> np.array:
        # новая матрица для дочерних нод
        reduced_matrix = self.matrix[:]

        new_row_indexes = self.row_indexes[:]
        new_column_indexes = self.column_indexes[:]

        reduced_matrix[column_i][row_i] = float('inf')

        del new_row_indexes[row_i]
        del new_column_indexes[column_i]

        # print(f'Удаляем:\nстроку {self.row_indexes[row_i]}'
        #       f'\nстолбец {self.column_indexes[column_i]}')

        reduced_matrix = np.delete(reduced_matrix, [row_i], 0)
        reduced_matrix = np.delete(reduced_matrix, [column_i], 1)

        return reduced_matrix, new_row_indexes, new_column_indexes

    def route_calculation(self):

        self.row_reduction()
        self.column_reduction()

        if self.root is not None:
            self.value += self.root.value

        row_index, column_index = self.evaluation_of_zero_cells()

        self.route = [self.row_indexes[row_index], self.column_indexes[column_index]]

        # for next node
        reduced_operation = self.matrix_reduce(row_index, column_index)

        self.next = Node(reduced_operation[0])

        self.next.new_node(self, reduced_operation[1], reduced_operation[2])

        if (self.route[0] in self.next.column_indexes) \
                and (self.route[1] in self.next.row_indexes):
            row_index = self.next.row_indexes.index(self.route[1])
            column_index = self.next.column_indexes.index(self.route[0])
            self.next.matrix[row_index][column_index] = float("inf")

        if len(self.matrix) > 1:
            self.next.route_calculation()
            self.check_parameters()

    def get_result(self):
        """ Выдать результат всего пути """

        if self.next is not None and len(self.next.matrix) != 1:
            return self.next.get_result()
        else:
            return self.value

    def get_route_node(self, route):
        """ Сбор пути по частям """
        route.append(self.route)
        if self.next is not None and len(self.next.matrix) != 0:
            route = self.next.get_route_node(route)

        return route

    def set_result(self, way):
        result_way = way[0]

        for i in range(len(way[1])):
            for j in range(2):
                set_way = way[1][i][j]
                check = True

                for k in range(len(result_way)):
                    if result_way[k] == set_way:
                        check = False
                if check:
                    result_way.insert(len(result_way) - 1, set_way)
        return result_way

    def get_result_way(self):
        """ Получить вектор опитмального пути  """
        all_ways = self.get_route_node(list())
        optimal_way = []

        for i in range(len(all_ways)):
            full_way = all_ways[:]
            optimal_way = [full_way[i][0], full_way[i][1]]
            del full_way[i]

            is_correct = True
            while len(full_way) != 0 and is_correct:
                flag = False
                for j in range(len(full_way)):
                    bunch = full_way[j]

                    if bunch[0] == optimal_way[len(optimal_way) - 1]:
                        optimal_way.append(bunch[1])
                        del full_way[j]
                        flag = True
                        break
                if not flag:
                    is_correct = False

        return [optimal_way, all_ways]


if __name__ == '__main__':
    import matrix_and_city as mc

    c0 = mc.City(0, 2)
    c1 = mc.City(2, 5)
    c2 = mc.City(5, 2)
    c3 = mc.City(6, 6)
    c4 = mc.City(8, 3)
    m = mc.Matrix([c0, c1, c2, c3, c4])

    root_node = Node(m.matrix, 0)
    root_node.route_calculation()

    root_node.next.route_calculation()

    way = root_node.get_result_way()[0]
    a = root_node.get_result()
    b = root_node.get_route_node([])
    get_res = root_node.get_result_way()

    print(get_res)

    # if len(way) != len(root_node.matrix) + 1:
    #     way = root_node.set_result(root_node.get_result_way())
    #
    #
    # del way[-1]
    # way = [x - 1 for x in way]
    #
    #
    # print(f'сумма маршрута: {root_node.get_result_way()}')
    # print(root_node.get_route_node([]))

# route1.row_reduction()
# route1.column_reduction()
# res = route1.evaluation_of_zero_cells()
# route1.matrix_reduce(res[0], res[1])
#
# print(route1.node.value)
