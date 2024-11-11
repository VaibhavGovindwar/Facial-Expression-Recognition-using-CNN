from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam

# Define LeNet-5 model
model = Sequential()

# First convolutional  layer
# 6 filters, each with a 5x5 kernel, ReLU activation function
# Input shape: (48, 48, 1) - input images of size 48x48 pixels with 1 channel (grayscale)
model.add(layers.Conv2D(6, (5, 5), activation='relu', input_shape=(48, 48, 1)))

# First max-pooling layer
# Max pooling with a 2x2 window size, reducing spatial dimensions by half
model.add(layers.MaxPooling2D((2, 2)))

# Second convolutional layer  
# 16 filters, each with a 5x5 kernel, ReLU activation function
model.add(layers.Conv2D(16, (5, 5), activation='relu'))

# Second max-pooling layer
# Max pooling with a 2x2 window size, reducing spatial dimensions by half
model.add(layers.MaxPooling2D((2, 2)))

# Flatten layer to convert 2D feature maps to 1D feature vectors
model.add(layers.Flatten())

# First fully connected layer
# 120 vectors with activation function ReLu 
model.add(layers.Dense(120, activation='relu'))

# Second fully connected layer
# 84 neurons with activation function ReLu 
model.add(layers.Dense(84, activation="relu"))
 
# Output layer
# 7 neurons for classification (7 emotion classes), softmax activation for multiclass classification
model.add(layers.Dense(7, activation='softmax'))

# Print model summary
model.summary()
