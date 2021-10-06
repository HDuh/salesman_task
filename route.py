from typing import List, Any

import numpy
import numpy as np
from math import sqrt
import matplotlib.pyplot as plt


class Point:
    """
    Класс адреса.
    Имеет координаты и название
    """
    counter = 0

    def __init__(self, x, y, address=None) -> None:
        self._x: int = x
        self._y: int = y
        Point.counter += 1
        if address is None:
            self.address = 'address' + str(Point.counter)
        else:
            self.address: str = address

    @property
    def x(self) -> int:
        """ Геттер координаты X """
        return self._x

    @property
    def y(self) -> int:
        """ Геттер координаты Y """
        return self._x

    @property
    def coordinate(self) -> tuple[int, int]:
        """ Геттер координат точки """
        return self._x, self._y


class Route:
    """
    Класс маршрута.
    Имеет перечень точек, по которым строится маршрут от почтового отделения по всем адресам.
    Умеет вычислять наикратчайший маршрут и строить график этого маршрута.
    """

    def __init__(self, addresses, start_point: int = 0) -> None:
        self._addresses: List[Point] = addresses

        self._total_distance: int = 0
        self.start_point = start_point
        self._route = list()
        self._route.append(start_point)

    @property
    def number_of_points(self) -> int:
        """
        Геттер.
        Кол-во адресов в маршруте.
        """
        return len(self._addresses)

    @property
    def route(self) -> Any:
        """
        Геттер.
        Построенный маршрут, если производилось вычисление.
        """
        if self._route != list([self.start_point]):
            return self._route
        else:
            raise Exception('Для просмотра маршрута '
                            'нужно выполнить метод "route_calculate".')

    @property
    def matrix(self) -> numpy.ndarray:
        """
        Геттер.
        Матрица кратчайших расстояний.
        """
        number: int = self.number_of_points
        matrix: np.array = np.zeros([number, number])
        for i in range(0, number):
            for j in range(0, number):
                if i == j:
                    # Заполнение главной диагонали матрицы
                    matrix[i, j] = float('inf')
                else:
                    matrix[i, j] = sqrt((self._addresses[i].coordinate[0] -
                                         self._addresses[j].coordinate[0]) ** 2 +
                                        (self._addresses[i].coordinate[1] -
                                         self._addresses[j].coordinate[1]) ** 2)

        return matrix

    def route_calculate(self) -> None:
        """
        Метод.
        Рассчитывает маршрут следования и его длину.
        """
        number = self.number_of_points
        matrix = self.matrix

        # Вычисление маршрута, основываясь на методе ближайшего соседа:
        for i in range(1, number):
            cur_row = list()
            for j in range(0, number):
                cur_row.append(matrix[self._route[i - 1], j])
            min_distance = min(cur_row)
            nearest_neighbors_i = cur_row.index(min_distance)  # индекс ближайшего адреса к текущей точке маршрута

            self._route.append(nearest_neighbors_i)

            for x in range(len(self._route)):
                for y in range(len(self._route) - 1):
                    matrix[self._route[x], self._route[y]] = float('inf')
            self._total_distance += min_distance

            print(f'Идём из точки {self._route[i - 1]} до {self._route[i]} -> {self._total_distance}')
            print(self._route[i - 1:i + 2])

        # возвращаемся в почтовое отделение и прибавляем путь из последней точки до почтового отделения
        self._route.append(self.start_point)
        self._total_distance += matrix[self._route[-1]][self._route[-2]]  # значение из матрицы расстояний

        print(f'Идём из точки {self._route[-2]} обратно в {self._route[-1]} -> {self._total_distance}')
        print(f'Маршрут: {self._route}')

    def route_graph(self) -> None:

        plt.title(f'Общий путь = {self._total_distance}\n'
                  f'Всего адресов: {self.number_of_points - 1}', size=15)
        X1 = [self._addresses[self._route[i]].coordinate[0] for i in range(0, self.number_of_points)]
        Y1 = [self._addresses[self._route[i]].coordinate[1] for i in range(0, self.number_of_points)]

        plt.plot(X1, Y1, color='red', linewidth=1)
        plt.plot(X1, Y1, color='black', linestyle=' ', marker='.')

        X2 = [self._addresses[self._route[-2]].coordinate[0], self._addresses[self._route[-1]].coordinate[0]]
        Y2 = [self._addresses[self._route[-2]].coordinate[1], self._addresses[self._route[-1]].coordinate[1]]
        plt.plot(X2, Y2, color='blue', linewidth=1, linestyle='--', label='Путь от последнего пункта до \n '
                                                                          'почтового отделения')
        plt.legend(loc='best')
        plt.grid(True)
        plt.show()


if __name__ == '__main__':
    # список адресов
    ADDRESSES = [Point(0, 2), Point(2, 5),
                 Point(5, 2), Point(6, 6),
                 Point(8, 3)]

    r = Route(ADDRESSES)
    # построить маршрут
    r.route_calculate()

    # нарисовать маршрут
    r.route_graph()
    print(r.matrix)