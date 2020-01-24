import argparse

import keras
import tensorflow as tf
from keras import backend as K

import cloudpickle

import mlflow
import mlflow.keras
import mlflow.pyfunc
from mlflow.pyfunc import PythonModel
from mlflow.utils.environment import _mlflow_conda_env

mlflow.start_run()

mlflow.keras.autolog()

parser = argparse.ArgumentParser(
    description='Train a Keras feed-forward network for MNIST classification in PyTorch')
parser.add_argument('--batch-size', '-b', type=int, default=128)
parser.add_argument('--epochs', '-e', type=int, default=4)
parser.add_argument('--learning-rate', '-l', type=float, default=0.05)
parser.add_argument('--num-hidden-units', '-n', type=int, default=512)
parser.add_argument('--dropout', '-d', type=float, default=0.25)
parser.add_argument('--momentum', '-m', type=float, default=0.85)
args = parser.parse_args()

mnist = keras.datasets.mnist
(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = keras.models.Sequential([
  keras.layers.Flatten(input_shape=x_train[0].shape),
  keras.layers.Dense(args.num_hidden_units, activation=tf.nn.relu),
  keras.layers.Dropout(args.dropout),
  keras.layers.Dense(10, activation=tf.nn.softmax)
])

optimizer = keras.optimizers.SGD(lr=args.learning_rate,
                                 momentum=args.momentum,
                                 nesterov=True)

model.compile(optimizer=optimizer,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train,
          epochs=args.epochs,
          batch_size=args.batch_size)

test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)

conda_env = _mlflow_conda_env(
    additional_conda_deps=[
        "keras=={}".format(keras.__version__),
        "tensorflow=={}".format(tf.__version__),
    ],
    additional_pip_deps=[
        "cloudpickle=={}".format(cloudpickle.__version__),
    ])

class KerasMnistCNN(PythonModel):

    def load_context(self, context):
        import tensorflow as tf
        self.graph = tf.Graph()
        with self.graph.as_default():
            K.set_learning_phase(0)
            self.model = mlflow.keras.load_model(context.artifacts["keras-model"])

    def predict(self, context, input_df):
        with self.graph.as_default():
            return self.model.predict(input_df.values.reshape(-1, 28, 28))

mlflow.pyfunc.log_model(
    artifact_path="model",
    python_model=KerasMnistCNN(),
    artifacts={
        "keras-model": mlflow.get_artifact_uri("model")
    },
    conda_env=conda_env)

mlflow.end_run()
