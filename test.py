import numpy as np

softmax_outputs = [[0.7, 0.2, 0.1],
                    [0.5, 0.1, 0.4],
                    [0.02, 0.9, 0.08]]

targets = [0, 1, 1]

predictions = np.argmax(softmax_outputs, axis=1)
accuracy = np.mean(predictions==targets)

print('Acc: ', accuracy)