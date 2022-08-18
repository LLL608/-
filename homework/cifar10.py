# %%
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt

# %%
cifar_10 = tf.keras.datasets.cifar10
(train_images, train_labels), (test_images, test_labels) = cifar_10.load_data()
print('Train: X=%s, y=%s' % (train_images.shape, train_labels.shape))
print('Test: X=%s, y=%s' % (train_images.shape, train_labels.shape))
# %%
# Inspect Images
from matplotlib import pyplot
for i in range(9):
	pyplot.subplot(330+1+i)  
	pyplot.imshow(train_images[i])
pyplot.show()
print(train_labels[:8])
# %%
# 归一化处理
from keras.utils import to_categorical
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)
train_images = train_images / 255.0
test_images = test_images / 255.0
print(train_labels[:8])
# %%
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
# %%
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
# %%
# Display the architecture of your model so far
model.summary()
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10))
model.summary()
# %%
# 训练模型
model.compile(optimizer='adam',
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=100, batch_size=64,validation_data=(train_images, train_labels))
  # %%
# 评估
loss, acc = model.evaluate(test_images, test_labels, verbose=0)
print('the accuracy of test set is: %.3f' % (acc * 100.0))

# %%
