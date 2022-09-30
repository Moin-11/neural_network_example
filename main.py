import numpy as np
import tensorflow as tf

# GRADED FUNCTION: house_model
def house_model():
    xs = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=float)
    ys = np.array([50000, 100000, 150000, 200000, 250000, 300000], dtype=float)

    model = tf.keras.Sequential([tf.keras.layers.Dense(units=1, input_shape=[1])])
    model.compile(optimizer='sgd', loss='mean_squared_error')
    model.fit(xs, ys, epochs=1000)

    return model


model = house_model()

new_y = 7.0
prediction = model.predict([new_y])[0]
print(prediction)

