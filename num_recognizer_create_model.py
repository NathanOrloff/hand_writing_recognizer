import tensorflow as tf
import tensorflow_datasets as tfds


INVERT = True

(ds_train, ds_test), ds_info = tfds.load(
    "mnist",
    split=["train", "test"],
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
)

def normalize_img(image, label):
    """Normalizes images: `uint8` -> `float32`."""
    return tf.cast(image, tf.float32) / 255., label

ds_train = ds_train.map(normalize_img)

if INVERT:
    def invert(image, label):
        return (tf.cast(image, tf.float32) * -1.0) + 1.0, label
    inverted = ds_train.map(invert)
    ds_train = ds_train.concatenate(inverted)

ds_train = ds_train.cache()
ds_train = ds_train.shuffle(ds_info.splits["train"].num_examples*(INVERT+1))
ds_train = ds_train.batch(128)
ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)


ds_test = ds_test.map(normalize_img)

if INVERT:
    inverted = ds_test.map(invert)
    ds_test = ds_test.concatenate(inverted)

ds_test = ds_test.batch(128)
ds_test = ds_test.cache()
ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)


max_pool = tf.keras.layers.MaxPool2D((2, 2), (2, 2), padding='same')

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(
        filters=16,
        kernel_size=5,
        padding="same",
        activation=tf.nn.relu),
    max_pool,
    tf.keras.layers.Conv2D(
        filters=32,
        kernel_size=5,
        padding="same",
        activation=tf.nn.relu),
    max_pool,
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(10, activation="softmax")
])

model.compile(
    loss="sparse_categorical_crossentropy",
    optimizer="adam",
    metrics=["accuracy"],
)

model.fit(
    ds_train,
    epochs=1,
    validation_data=ds_test,
)


model.save('my_model')

