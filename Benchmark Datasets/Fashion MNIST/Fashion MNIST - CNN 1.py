import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers



# Checks hardware status to see if GPUs are available for training.
print("Checking system hardwares ...")
print("No. of CPUs available: %d" %(len(tf.config.list_physical_devices(("CPU")))))
print("No. of GPUs available: %d" %(len(tf.config.list_physical_devices(("GPU")))))



# Loads train and test data in the original dataset and displays the dimensions.
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
print("Train: x = %s, y = %s" %(x_train.shape, y_train.shape))
print("Test: x = %s, y = %s" %(x_test.shape, y_test.shape))

x_train = x_train.astype("float32")
x_test = x_test.astype("float32")



# DATA PREPROCESSING.
# Data is preprocessed by min-max convention so that all preprocessed pixel values lie between zero and one.
# Compared to direct division by 255, min-max convention should be theoretically more robust,
# since preprocessed values are invariant under simultaneous addition and multiplication of original pixel values,
# thus providing some standardization for images under different light conditions and contrasts.
for n in range(0, x_train.shape[0]):
    max_pixel = x_train[n,:,:].max()
    min_pixel = x_train[n,:,:].min()
    x_train[n,:,:] = (max_pixel - x_train[n,:,:]) / (max_pixel - min_pixel)

for n in range(0, x_test.shape[0]):
    max_pixel = x_test[n,:,:].max()
    min_pixel = x_test[n,:,:].min()
    x_test[n,:,:] = (max_pixel - x_test[n,:,:]) / (max_pixel - min_pixel)

print("Value range in preprocessed train set: [%f, %f]." %(x_train.min(), x_train.max()))
print("Value range in preprocessed test set: [%f, %f]." %(x_test.min(), x_test.max()))

# Prepares data for model input.
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)



# Compiles, trains and evaluates the CNN model.
model = Sequential([
    layers.Conv2D(filters = 32, kernel_size = (1, 5), strides = 1, padding = "same",
        activation = layers.ReLU()),
    layers.Conv2D(filters = 32, kernel_size = (5, 1), strides = 1, padding = "same",
        activation = layers.ReLU()),
    layers.Conv2D(filters = 64, kernel_size = (3, 3), strides = 1, padding = "same",
        activation = layers.ReLU()),
    layers.MaxPool2D(pool_size = (2, 2)),
    layers.Flatten(),
    layers.Dense(128,
        activation = layers.ReLU()),
    layers.Dropout(0.5),
    layers.Dense(10,
        activation = layers.Softmax())
])
model.compile(
    optimizer = "adam",
    loss = "sparse_categorical_crossentropy",
    metrics = ["accuracy"]
)
model.fit(x = x_train, y = y_train,
    epochs = 10)

model.evaluate(x = x_test, y = y_test)

model.summary()