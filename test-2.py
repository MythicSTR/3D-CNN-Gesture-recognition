import tensorflow as tf
import numpy as np
import cv2

# Define gesture classes
classes = [
    "Swiping Right",
    "Sliding Two Fingers Left",
    "No gesture",
    "Thumb Up"
]

def normalize_data(np_data):
    # Normalize pixel values
    np_data = np_data.astype(np.float32) / 255.0  # Normalize to [0, 1]
    return np_data.reshape(30, 64, 64, 1)  # Shape should be (30, 64, 64, 1)

class Conv3DModel(tf.keras.Model):
    def __init__(self):
        super(Conv3DModel, self).__init__()
        # Convolutions
        self.conv1 = tf.keras.layers.Conv3D(32, (3, 3, 3), activation='relu', input_shape=(30, 64, 64, 1), name="conv1")
        self.pool1 = tf.keras.layers.MaxPool3D(pool_size=(2, 2, 2))
        self.conv2 = tf.keras.layers.Conv3D(64, (3, 3, 3), activation='relu', name="conv2")
        self.pool2 = tf.keras.layers.MaxPool3D(pool_size=(2, 2, 2))
        self.convLSTM = tf.keras.layers.ConvLSTM2D(40, (3, 3), return_sequences=False)
        self.flatten = tf.keras.layers.Flatten()
        self.d1 = tf.keras.layers.Dense(128, activation='relu')
        self.out = tf.keras.layers.Dense(len(classes), activation='softmax')

    def call(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.convLSTM(x)
        x = self.flatten(x)
        x = self.d1(x)
        return self.out(x)

# Initialize model
new_model = Conv3DModel()

# Create a checkpoint object
checkpoint = tf.train.Checkpoint(model=new_model)

# Restore the weights
checkpoint.restore('weights/path_to_my_weights2').expect_partial()

to_predict = []
cap = cv2.VideoCapture(0)
classe = ''  # Initialize the variable

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to grayscale and resize
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized_frame = cv2.resize(gray, (64, 64))
    to_predict.append(resized_frame)

    if len(to_predict) == 30:
        frame_to_predict = np.array(to_predict, dtype=np.float32)
        frame_to_predict = normalize_data(frame_to_predict)  # Normalize the data

        # Make prediction
        predict = new_model.predict(np.expand_dims(frame_to_predict, axis=0))  # Add batch dimension
        classe = classes[np.argmax(predict)]
        
        print('Class =', classe, 'Precision =', np.amax(predict) * 100, '%')

        # Reset for next prediction
        to_predict = []

    # Display the predicted class on the video feed
    cv2.putText(frame, classe if classe else 'Waiting for gesture...', (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 1, cv2.LINE_AA)
    
    cv2.imshow('Hand Gesture Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
