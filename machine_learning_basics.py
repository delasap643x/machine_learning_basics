import numpy as np
from keras.models import Sequential
from keras.layers import Dense

"""
Generate some dummy data for training
"""
X_train = np.random.rand(100, 2)  # Input
y_train = np.random.rand(100, 2)  # Output

"""
Create a feedforward neural network model
"""
model = Sequential()
model.add(Dense(64, input_dim=2, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(2, activation='linear'))    #position detection

model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

"""Training"""
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

"""Use the trained model for position detection"""
X_test = np.array([[0.5, 0.3]])  # Input features for position detection
predicted_position = model.predict(X_test)
print("Predicted position:", predicted_position)