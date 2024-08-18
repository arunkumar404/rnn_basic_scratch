import numpy as np
import random

from rnn import RNN
from data import train_data, test_data

vocab = list(set([w for text in train_data.keys() for w in text.split(' ')]))
vocab_size = len(vocab)
print('%d unique words found' % vocab_size)

word_to_idx = { w: i for i, w in enumerate(vocab) }
idx_to_word = { i: w for i, w in enumerate(vocab) }

def createInputs(text):
  inputs = []
  for w in text.split(' '):
    v = np.zeros((vocab_size, 1))
    v[word_to_idx[w]] = 1
    inputs.append(v)
  return inputs

def softmax(xs):
  return np.exp(xs) / sum(np.exp(xs))

rnn = RNN(vocab_size, 2)

def processData(data, backprop=True):
  items = list(data.items())
  random.shuffle(items)

  loss = 0
  num_correct = 0

  for x, y in items:
    inputs = createInputs(x)
    target = int(y)

    out, _ = rnn.forward(inputs)
    probs = softmax(out)

    loss -= np.log(probs[target])
    num_correct += int(np.argmax(probs) == target)

    if backprop:
      dL_dy = probs
      dL_dy[target] -= 1

      rnn.backprop(dL_dy)

  return loss / len(data), num_correct / len(data)

for epoch in range(1000):
  train_loss, train_acc = processData(train_data)

  if epoch % 100 == 99:
    print('--- Epoch %d' % (epoch + 1))
    print('Train:\tLoss %.3f | Accuracy: %.3f' % (train_loss, train_acc))

    test_loss, test_acc = processData(test_data, backprop=False)
    print('Test:\tLoss %.3f | Accuracy: %.3f' % (test_loss, test_acc))
