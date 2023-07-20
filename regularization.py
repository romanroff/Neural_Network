# ------------------------ MNIST ------------------------ #
# import sys
# import numpy as np
# from keras.datasets import mnist
#
# (x_train, y_train), (x_test, y_test) = mnist.load_data()
#
# images, labels = (x_train[:1000].reshape(1000, 28*28) / 255, y_train[:1000])
# one_shot_labels = np.zeros((len(labels), 10))
# for i, l in enumerate(labels):
#     one_shot_labels[i][l] = 1
# labels = one_shot_labels
#
# test_images = x_test.reshape(len(x_test), 28*28) / 255
# test_labels = np.zeros((len(y_test), 10))
# for i, l in enumerate(y_test):
#     test_labels[i][l] = 1
#
# np.random.seed(1)
# relu = lambda x: (x > 0) * x
# relu2rediv = lambda x: x >= 0
# alpha, iterations, hidden_size, pixels_per_image, num_labels = (.005, 300, 100, 784, 10)
# weight01 = .2 * np.random.random((pixels_per_image, hidden_size)) - .1
# weight12 = .2 * np.random.random((hidden_size, num_labels)) - .1
#
# for j in range(iterations):
#     error, correct_cnt = (.0, 0)
#
#     for i in range(len(images)):
#         layer_0 = images[i:i+1]
#         layer_1 = relu(np.dot(layer_0, weight01))
#         dropout_mask = np.random.randint(2, size=layer_1.shape)
#         layer_1 *= dropout_mask * 2
#         layer_2 = np.dot(layer_1, weight12)
#         error += np.sum((labels[i:i+1] - layer_2) ** 2)
#         correct_cnt += int(np.argmax(layer_2) == np.argmax(labels[i:i+1]))
#         layer_2_delta = labels[i:i+1] - layer_2
#         layer_1_delta = layer_2_delta.dot(weight12.T) * relu2rediv(layer_1)
#         layer_1_delta *= dropout_mask
#         weight12 += alpha * layer_1.T.dot(layer_2_delta)
#         weight01 += alpha * layer_0.T.dot(layer_1_delta)
#
#     if j % 10 == 0:
#         test_error, test_correct_cnt = (.0, 0)
#         for i in range(len(test_images)):
#             layer_0 = test_images[i:i+1]
#             layer_1 = relu(np.dot(layer_0, weight01))
#             layer_2 = np.dot(layer_1, weight12)
#             test_error += np.sum((test_labels[i:i+1] - layer_2) ** 2)
#             test_correct_cnt += int(np.argmax(layer_2) == np.argmax(test_labels[i:i+1]))
#         sys.stdout.write(f"\nI: {j} "
#                          f"Test-Err: {test_error / float(len(test_images)):.3f} "
#                          f"Test-Acc: {test_correct_cnt / float(len(test_images))} "
#                          f"Train-Err: {error / float(len(images)):.3f} "
#                          f"Train-Acc: {correct_cnt / float(len(images))} ")

# ------------------------ Пакетный градиентный спуск ------------------------ #
# import sys
# import numpy as np
# from keras.datasets import mnist
# np.random.seed(1)
#
# (x_train, y_train), (x_test, y_test) = mnist.load_data()
# images, labels = (x_train[:1000].reshape(1000, 28*28) / 255, y_train[:1000])
# one_shot_labels = np.zeros((len(labels), 10))
# for i, l in enumerate(labels):
#     one_shot_labels[i][l] = 1
# labels = one_shot_labels
#
# test_images = x_test.reshape(len(x_test), 28*28) / 255
# test_labels = np.zeros((len(y_test), 10))
# for i, l in enumerate(y_test):
#     test_labels[i][l] = 1
#
#
# def relu(x):
#     return (x >= 0) * x
#
#
# def relu2dev(x):
#     return x >= 0
#
#
# batch_size = 100
# alpha, iterations = (0.001, 300)
# pixels_per_image, num_labels, hidden_size = (784, 10, 100)
# weight01 = .2 * np.random.random((pixels_per_image, hidden_size)) - .1
# weight12 = .2 * np.random.random((hidden_size, num_labels)) - .1
#
# for j in range(iterations):
#     error, correct_cnt = (0.0, 0)
#     for i in range(int(len(images) / batch_size)):
#         batch_start, batch_end = ((i * batch_size), ((i + 1) * batch_size))
#
#         layer_0 = images[batch_start:batch_end]
#         layer_1 = relu(layer_0.dot(weight01))
#         dropout_mask = np.random.randint(2, size=layer_1.shape)
#         layer_1 *= dropout_mask * 2
#         layer_2 = layer_1.dot(weight12)
#
#         error += np.sum((labels[batch_start:batch_end] - layer_2) ** 2)
#         for k in range(batch_size):
#             correct_cnt += int(np.argmax(layer_2[k:k+1]) == np.argmax(labels[batch_start+k:batch_start+k+1]))
#
#             layer_2_delta = (labels[batch_start:batch_end] - layer_2) / batch_size
#             layer_1_delta = layer_2_delta.dot(weight12.T) * relu2dev(layer_1)
#             layer_1_delta *= dropout_mask
#
#             weight12 += alpha * layer_1.T.dot(layer_2_delta)
#             weight01 += alpha * layer_0.T.dot(layer_1_delta)
#     if j % 10 == 0:
#         test_error = 0.0
#         test_correct_cnt = 0
#         for i in range(len(test_images)):
#             layer_0 = test_images[i:i+1]
#             layer_1 = relu(layer_0.dot(weight01))
#             layer_2 = layer_1.dot(weight12)
#             test_error += np.sum((test_labels[i:i+1] - layer_2) ** 2)
#             test_correct_cnt += int(np.argmax(layer_2) == np.argmax(test_labels[i:i+1]))
#         sys.stdout.write(f"\nI: {j} "
#                          f"Test-Err: {test_error / float(len(test_images)):.3f} "
#                          f"Test-Acc: {test_correct_cnt / float(len(test_images))} "
#                          f"Train-Err: {error / float(len(images)):.3f} "
#                          f"Train-Acc: {correct_cnt / float(len(images))} ")

# ------------------------ Усовершенствование сети MNIST с помощью softmax ------------------------ #
# import numpy as np, sys
# np.random.seed(1)
# from keras.datasets import mnist
#
# (x_train, y_train), (x_test, y_test) = mnist.load_data()
#
# images, labels = (x_train[0:1000].reshape(1000,28*28) / 255, y_train[0:1000])
#
# one_hot_labels = np.zeros((len(labels),10))
# for i,l in enumerate(labels):
#     one_hot_labels[i][l] = 1
# labels = one_hot_labels
#
# test_images = x_test.reshape(len(x_test),28*28) / 255
# test_labels = np.zeros((len(y_test),10))
# for i,l in enumerate(y_test):
#     test_labels[i][l] = 1
#
#
# def tanh(x):
#     return np.tanh(x)
#
#
# def tanh2deriv(x):
#     return 1 - (x ** 2)
#
#
# def softmax(x):
#     temp = np.exp(x)
#     return temp / np.sum(temp, axis=1, keepdims=True)
#
#
# alpha, iterations, hidden_size = (2, 300, 100)
# pixels_per_image, num_labels = (784, 10)
# batch_size = 100
# weight01 = .02 * np.random.random((pixels_per_image, hidden_size)) - .01
# weight12 = .2 * np.random.random((hidden_size, num_labels)) - .1
#
# for j in range(iterations):
#     correct_cnt = 0
#     for i in range(int(len(images) / batch_size)):
#         batch_start, batch_end = ((i * batch_size), ((i + 1) * batch_size))
#         layer_0 = images[batch_start:batch_end]
#         layer_1 = tanh(np.dot(layer_0, weight01))
#         dropout_mask = np.random.randint(2, size=layer_1.shape)
#         layer_1 *= dropout_mask * 2
#         layer_2 = softmax(np.dot(layer_1, weight12))
#
#         for k in range(batch_size):
#             correct_cnt += int(np.argmax(layer_2[k:k + 1]) == np.argmax(labels[batch_start + k:batch_start + k + 1]))
#
#         layer_2_delta = (labels[batch_start:batch_end] - layer_2) / (batch_size * layer_2.shape[0])
#         layer_1_delta = np.dot(layer_2_delta, weight12.T) * tanh2deriv(layer_1)
#         layer_1_delta *= dropout_mask
#
#         weight12 += alpha * layer_1.T.dot(layer_2_delta)
#         weight01 += alpha * layer_0.T.dot(layer_1_delta)
#
#     test_correct_cnt = 0
#
#     for i in range(len(test_images)):
#         layer_0 = test_images[i:i+1]
#         layer_1 = tanh(np.dot(layer_0, weight01))
#         layer_2 = np.dot(layer_1, weight12)
#
#         test_correct_cnt += int(np.argmax(layer_2) == np.argmax(test_labels[i:i + 1]))
#     if j % 10 == 0:
#         sys.stdout.write(f'\n'
#                          f'I: {j} Test-Acc: {test_correct_cnt / float(len(test_images))} '
#                          f'Train-Acc: {correct_cnt / float(len(images))}')