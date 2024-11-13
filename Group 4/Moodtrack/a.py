from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

model = Sequential([
    Dense(10, activation='relu', input_shape=(5,)),
    Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy')
model.save('test_model.h5')

# Load the dummy model
from tensorflow.keras.models import load_model
loaded_model = load_model('test_model.h5')
