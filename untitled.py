import glob
import hashlib

import matplotlib.pyplot as plt
import neptune.new as neptune
import numpy as np
import pandas as pd
import tensorflow as tf
from neptune.new.integrations.tensorflow_keras import NeptuneCallback
from scikitplot.metrics import plot_roc, plot_precision_recall

# Select project
run = neptune.init(project='common/example-project-tensorflow-keras',
                   tags=['mri', 'svm'],
                   name='keras-training')

# Prepare params
parameters = {'dense_units': 64,
              'activation': 'relu',
              'dropout': 0.3,
              'learning_rate': 0.09,
              'batch_size': 64,
              'n_epochs': 30}

run['model/params'] = parameters

# Prepare dataset
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
x_train = x_train / 255.0
x_test = x_test / 255.0

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Log data version
run['data/version/x_train'] = hashlib.md5(x_train).hexdigest()
run['data/version/y_train'] = hashlib.md5(y_train).hexdigest()
run['data/version/x_test'] = hashlib.md5(x_test).hexdigest()
run['data/version/y_test'] = hashlib.md5(y_test).hexdigest()
run['data/class_names'] = class_names

# Log example images
for j, class_name in enumerate(class_names):
    plt.figure(figsize=(10, 10))
    label_ = np.where(y_train == j)
    for i in range(9):
        plt.subplot(3, 3, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(x_train[label_[0][i]], cmap=plt.cm.binary)
        plt.xlabel(class_names[j])
    run['data/train_sample'].log(neptune.types.File.as_image(plt.gcf()))
    plt.close('all')

# Prepare model
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(parameters['dense_units'], activation=parameters['activation']),
    tf.keras.layers.Dropout(parameters['dropout']),
    tf.keras.layers.Dense(parameters['dense_units'], activation=parameters['activation']),
    tf.keras.layers.Dropout(parameters['dropout']),
    tf.keras.layers.Dense(10, activation='softmax')
])
optimizer = tf.keras.optimizers.SGD(learning_rate=parameters['learning_rate'])
model.compile(optimizer=optimizer,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Log model summary
model.summary(print_fn=lambda x: run['model/summary'].log(x))

# Train model
neptune_cbk = NeptuneCallback(run=run, base_namespace='metrics')

model.fit(x_train, y_train,
          batch_size=parameters['batch_size'],
          epochs=parameters['n_epochs'],
          validation_split=0.2,
          callbacks=[neptune_cbk])

# Log model weights
model.save('trained_model')
run['model/weights/saved_model'].upload('trained_model/saved_model.pb')
for name in glob.glob('trained_model/variables/*'):
    run[name].upload(name)

# Evaluate model
eval_metrics = model.evaluate(x_test, y_test, verbose=0)
for j, metric in enumerate(eval_metrics):
    run['test/scores/{}'.format(model.metrics_names[j])] = metric

# Log predictions as table
y_pred_proba = model.predict(x_test)
y_pred = np.argmax(y_pred_proba, axis=1)
y_pred = y_pred
df = pd.DataFrame(data={'y_test': y_test, 'y_pred': y_pred, 'y_pred_probability': y_pred_proba.max(axis=1)})
run['test/predictions'] = neptune.types.File.as_html(df)

# Log model performance visualizations
fig, ax = plt.subplots()
plot_roc(y_test, y_pred_proba, ax=ax)
run['charts/ROC'] = neptune.types.File.as_image(fig)

fig, ax = plt.subplots()
plot_precision_recall(y_test, y_pred_proba, ax=ax)
run['charts/precision-recall'] = neptune.types.File.as_image(fig)
plt.close('all')

run.wait()
