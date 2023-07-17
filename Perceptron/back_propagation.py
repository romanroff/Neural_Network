# ------------------------ Создание нейронной сети ------------------------ #
# import numpy as np
# weights = np.array([0.5,0.48,-0.7])
# alpha = 0.1
# streetlights = np.array([[1, 0, 1],
#                          [0, 1, 1],
#                          [0, 0, 1],
#                          [1, 1, 1],
#                          [0, 1, 1],
#                          [1, 0, 1]])
# walk_vs_stop = np.array([0, 1, 0, 1, 1, 0])
# input = streetlights[0]
# goal_pred = walk_vs_stop[0]
# for i in range(20):
#     pred = input.dot(weights)
#     error = (pred - goal_pred) ** 2
#     delta = pred - goal_pred
#     weights -= delta * input * alpha
#     print("Error:" + str(error) + " Prediction:" + str(pred))

# ------------------------ Обучение на полном наборе данных ------------------------ #
# import numpy as np
# weights = np.array([0.5,0.48,-0.7])
# alpha = 0.1
# streetlights = np.array([[1, 0, 1],
#                          [0, 1, 1],
#                          [0, 1, 1],
#                          [1, 0, 0],
#                          [1, 0, 1],
#                          [1, 1, 0]])
# walk_vs_stop = np.array([0, 1, 1, 0, 0, 1])
# for i in range(31):
#     error_for_all_lights = 0
#     for row_index in range(len(walk_vs_stop)):
#         input = streetlights[row_index]
#         goal_pred = walk_vs_stop[row_index]
#         pred = input.dot(weights)
#         error = (goal_pred - pred) ** 2
#         error_for_all_lights += error
#         delta = pred - goal_pred
#         weights -= alpha * input * delta
#         print(f"Prediction: {pred:.2f}")
#     print(f"Error: {error_for_all_lights:.5f}\n")

# ------------------------ Первая глубокая нейронная сеть ------------------------ #
# import numpy as np
# np.random.seed(1)
#
#
# def relu(x):
#     return (x > 0) * x                              # если x < 0, то будет (False * x) или (0 * x)
#
#
# alpha = .2
# hidden_size = 4
# streetlights = np.array([[ 1, 0, 1],
#                          [ 0, 1, 1],
#                          [ 0, 0, 1],
#                          [ 1, 1, 1 ]])
# walk_vs_stop = np.array([1, 1, 0, 0]).T
# weights_0_1 = 2 * np.random.random((3, hidden_size)) - 1
# weights_1_2 = 2 * np.random.random((hidden_size, 1)) - 1
# layer_0 = streetlights[0]
# layer_1 = relu(np.dot(layer_0, weights_0_1))
# layer_2 = np.dot(layer_1, weights_1_2)
# print(layer_2)

# ------------------------ Обратное распространение в коде ------------------------ #
import numpy as np                                                              # импорт, что логично
np.random.seed(1)                                                               # установка сида,
binary = np.array([[0,0,1],                                                     # чтобы менять гиперпараметры
                   [0,1,0],                                                     # и видеть разницу, что невозможно с
                   [1,1,0],                                                     # разными начальными весами
                   [1,0,1]])                                                    #
goal =np.array([1,0,0,1]).T                                                     #
alpha = .1                                                                      #
hidden_size = 4                                                                 # количество узлов (нейронов) в слое
weights01 = np.random.random((3, hidden_size))                                  # установка начальных весов
weights12 = np.random.random((hidden_size, 1))                                  # установка начальных весов


def relu(x):                                                                    # функция relu, чтобы отключить
    return (x > 0) * x                                                          # отрицательные веса


def relu2rediv(x):                                                              # функция, чтобы не менять параметры
    return x > 0                                                                # выключенных весов


epochs = 0                                                                      # количество необходимых эпох для обучения
while True:                                                                     # запуск обучения
    epochs += 1                                                                 # увеличение количества эпох на 1
    error_all = 0                                                               # сумма всех ошибок
    for i in range(len(binary)):                                                # проход по каждому входному набору
        layer_0 = binary[i:i+1]                                                 # входной набор
        layer_1 = relu(np.dot(layer_0, weights01))                              # параметры скрытого слоя
        layer_2 = layer_1.dot(weights12)                                        # результат предсказания
        error_all = np.sum((goal[i:i+1] - layer_2) ** 2)                        # сумма квадратичных ошибок
        error_2_delta = goal[i:i+1] - layer_2                                   # отклонение от истинного результата
        error_1_delta = error_2_delta.dot(weights12.T) * relu2rediv(layer_1)    # расчет отклонения для скрытого слоя
        weights12 += alpha * (layer_1.T.dot(error_2_delta))                     # изменение весов
        weights01 += alpha * (layer_0.T.dot(error_1_delta))                     # изменение весов
        print(f'Layer{i}: {layer_2[0][0]:.9f}')                                 # вывод результата по входным параметрам
    print(f"Error: {error_all:.10f}\n")                                         # вывод суммы ошибок
    if round(error_all, 9) < 10e-10:                                            # остановка обучения, если ошибка
        break                                                                   # меньше целевого значения
print(f"Epochs: {epochs}")                                                      # вывод результата необходимых эпох


















