import pickle
from local_code.stage_3_code.Method_CNN_MNIST import Method_CNN_MNIST

with open('/Users/mako/PycharmProjects/ECS_170_Spring_2026_Project/data/stage_3_data/MNIST', 'rb') as f:
    data = pickle.load(f)

method = Method_CNN_MNIST('cnn_mnist', 'CNN for mnist')
method.data = data

result = method.run()

correct = sum(p == t for p, t in zip(result['pred_y'], result['true_y']))
total   = len(result['true_y'])
print(f'Accuracy: {100 * correct / total:.2f}%')