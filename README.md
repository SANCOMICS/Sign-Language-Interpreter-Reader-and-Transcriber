# Sign-Language-Interpreter-Reader-and-Transcriber
a sign language interpreter that uses the device camera to capture a sign language user then transcribes the signs into text then reads the words created using a text to speech module



# KEYPOINT CLASSIFICATION IPYNB


1. Load and Preprocess the Data:
Dataset: Reads gesture/keypoint data from keypoint.csv. Each sample consists of:

Features: 42 numerical values (representing 21 2D keypoints, e.g., from a hand tracking model like MediaPipe).
Labels: A single integer (the gesture class label, ranging from 0 to 25 for 26 classes, assuming A-Z gestures).
Invalid Label Check: Identifies and prints labels that exceed the defined NUM_CLASSES (26).

Train-Test Split: Splits the dataset into:

X_train, X_test (features for training and testing)
y_train, y_test (labels for training and testing)
Ensures 75% of the data is used for training.
2. Define and Compile the Model:
Architecture: A simple feed-forward neural network:

Input Layer: 42 features (21 keypoints × 2 dimensions).
Hidden Layers:
Two Dense layers with 20 and 10 units respectively, using ReLU activation.
Dropout layers (20% and 40%) to reduce overfitting.
Output Layer: A Dense layer with 26 units, using a softmax activation to predict gesture classes (one-hot probabilities).
Compilation:

Loss Function: sparse_categorical_crossentropy for multi-class classification.
Optimizer: Adam optimizer for efficient training.
Metrics: Tracks accuracy during training and evaluation.
3. Train the Model:
Callbacks:

Checkpoint Callback: Saves the best model during training to keypoint_classifier.keras.
Early Stopping: Stops training if validation performance does not improve for 20 consecutive epochs.
Training:

Trains the model for up to 1000 epochs with a batch size of 128.
Validates the model on the test set (X_test, y_test) after each epoch.
4. Evaluate the Model:
Evaluation Metrics:

Computes the validation loss (val_loss) and accuracy (val_acc) on the test set.
Predictions:

Makes predictions on a single test sample and prints:
The predicted class probabilities.
The most probable class (predicted label).
5. Convert to TensorFlow Lite (TFLite):
TFLite Conversion:

Converts the trained model into a lightweight TensorFlow Lite model (keypoint_classifier.tflite) for deployment on devices with limited resources (e.g., mobile or embedded systems).
Enables quantization to reduce model size and improve efficiency.
TFLite Testing:

Loads the TFLite model into an interpreter.
Runs inference on a single test sample and prints:
The class probabilities.
The predicted label.



# POINT HISTORY CLASSIFICATION IPYNB

Load Gesture Data:

Reads a dataset of gestures (point_history.csv), where each sample represents a sequence of 16 timesteps with 2D coordinates and a label.
Preprocess Data:

Splits the data into training and testing sets for training and evaluation.
Build and Train a Gesture Recognition Model:

Defines a neural network for gesture classification, either using:
An LSTM-based model for sequential data, or
A Dense-layer-based model for simpler processing.
Compiles the model with Adam optimizer and sparse categorical cross-entropy loss.
Trains the model with callbacks for checkpointing and early stopping.
Evaluate the Model:

Tests the trained model on unseen data and generates predictions.
Displays the results using a confusion matrix and classification report to measure accuracy and performance.
Convert the Model to TensorFlow Lite (TFLite):

Converts the trained model into a lightweight format (TFLite) with quantization for deployment in resource-constrained environments (like mobile devices).
Run Inference with TFLite:

Loads the TFLite model into an interpreter and tests it by running predictions on new data to ensure it works as expected.







# APP PY


Command-line Argument Parsing:

Allows setting device, video dimensions, and detection/tracking confidence levels.
--use_static_image_mode: Toggles between static and dynamic image processing.
Camera Configuration:

Captures live video feed from a specified device.
Sets the resolution of the video feed.
Hand Gesture Detection:

Uses MediaPipe's Hands solution to detect hand landmarks.
Identifies gestures using:
KeyPointClassifier: Recognizes static hand gestures.
PointHistoryClassifier: Analyzes hand movement history.
Gesture Mapping and Processing:

Maps gestures to actions or letters based on predefined labels.
Supports "SPACE," "DELETE," and "READ" gestures:
Adds spaces, deletes characters, and reads the constructed word aloud using a text-to-speech engine.
Prediction Queue and Time Window:

Maintains a 2-second prediction window to determine the most common gesture.
Updates the current word based on the detected gesture.
Real-time Feedback:

Displays recognized gestures, bounding boxes, and the constructed word on the video feed.
Utilizes OpenCV's text rendering with a transparent background for better visualization.
Customizable Modes:

Supports switching between modes using numeric keys.
Maps keyboard keys ('a'-'z') to corresponding numerical values for gesture classification.
Visualization Enhancements:

Draws bounding boxes around detected hands.
Shows hand landmarks and gesture information on the screen.


        In app.py, the different modes are controlled by the command-line arguments and can be switched using specific options

        1. Static vs. Dynamic Image Processing
        --use_static_image_mode: This argument toggles between static and dynamic image processing.
        Static Mode: The program processes a single static image (frame) at a time. This is useful when you want to test or process a single gesture.
        Dynamic Mode: The program continuously processes frames from a video feed (real-time processing). This mode is ideal for detecting gestures in a live video stream.
        2. Camera Configuration
        Device and Video Settings: The app allows the configuration of the video feed using command-line arguments that can define:
        The specific device (camera) used for capturing video.
        The video resolution, which determines the quality and size of the captured frames.
        3. Gesture Detection Modes
        KeyPointClassifier Mode: This mode uses a static hand gesture classifier (based on key points) to recognize hand gestures when they are stationary. It's for detecting individual gestures, such as specific letters or symbols made by the hand.
        PointHistoryClassifier Mode: This mode analyzes hand movement over time and recognizes gestures based on the sequence of hand movements (gesture history). It's ideal for recognizing gestures that require movement or transitions, not just static positions.
        4. Gesture Mapping and Actions
        SPACE, DELETE, and READ Gestures: These special gestures are mapped to actions:
        SPACE: Adds a space character in the recognized text.
        DELETE: Deletes the last character typed or removes the last gesture input.
        READ: Converts the typed text into speech (text-to-speech), allowing the user to hear the transcription.
        5. Prediction Queue and Time Window
        2-second prediction window: The app uses a 2-second time window to decide the most probable gesture, ensuring that a single gesture is detected correctly over a short period of time.
        Updates Current Word: Based on the most common gesture in the last 2 seconds, it updates the current word being typed.
        6. Real-time Feedback Mode
        Visual Feedback: This mode provides real-time feedback by:
        Drawing bounding boxes around the detected hands.
        Displaying hand landmarks on the video feed.
        Showing the recognized gesture and constructed word on the screen.
        7. Customizable Mode Switching
        Mode Switching: The app allows switching between different modes (KeyPointClassifier and PointHistoryClassifier) using numeric keys. Each key corresponds to a specific classifier or processing mode. This flexibility allows users to experiment with different methods for gesture recognition.
        8. Visualization Enhancements
        The app enhances the visualization of hand gestures by drawing:
        Bounding boxes around the detected hands.
        Hand landmarks and gesture information to improve the user experience and understanding of the process.





# KEYPOINT CLASSIFIER PY

Initialization (__init__ method):

model_path: Specifies the path to the TFLite model file (default: 'model/keypoint_classifier/keypoint_classifier.tflite').
num_threads: Specifies the number of threads to use for inference (default: 1).
Loads the TFLite model using the TensorFlow Lite interpreter (tf.lite.Interpreter).
Allocates tensors for input and output data using self.interpreter.allocate_tensors().
Retrieves information about the input and output tensors:
self.input_details: Input tensor details.
self.output_details: Output tensor details.
Inference (__call__ method):

Takes a list of keypoints (landmark_list) as input.
Prepares the input tensor:
Retrieves the input tensor's index (self.input_details[0]['index']).
Converts the input data (landmark_list) to a NumPy array of type float32 and sets it as the input tensor.
Runs the model inference with self.interpreter.invoke().
Retrieves the output tensor:
Finds the output tensor's index (self.output_details[0]['index']).
Gets the output tensor using self.interpreter.get_tensor(...).
Processes the output:
Uses np.argmax to get the index of the maximum value in the output tensor, indicating the predicted class.



Output:

Returns the predicted class index (result_index) as the classification result.




# POINT HISTORY CLASSIFIER PY



Initialization (__init__ method):
Arguments:
model_path: Path to the TFLite model (default: 'model/point_history_classifier/point_history_classifier.tflite').
score_th: Confidence score threshold for classification (default: 0.5). If the predicted score is below this threshold, the result is considered invalid.
invalid_value: Value to return for invalid classifications (default: 0).
num_threads: Number of threads for inference (default: 1).
Setup:
Loads the TFLite model using TensorFlow Lite's tf.lite.Interpreter.
Allocates tensors using self.interpreter.allocate_tensors().
Retrieves input and output tensor details:
self.input_details: Information about input tensor.
self.output_details: Information about output tensor.
Stores the confidence threshold (score_th) and invalid value (invalid_value).
2. Inference (__call__ method):
Input:
Takes a list of point history data (point_history), which could represent a sequence of coordinates or data points over time.
Steps:
Prepare Input:
Retrieves the input tensor's index (self.input_details[0]['index']).
Converts the input data (point_history) to a NumPy array of type float32 and sets it as the input tensor.
Run Inference:
Calls self.interpreter.invoke() to perform model inference.
Retrieve Output:
Gets the output tensor's index (self.output_details[0]['index']).
Retrieves the model's prediction result using self.interpreter.get_tensor(...).
Post-Processing:
Finds the index of the maximum score in the output using np.argmax, which represents the predicted class (result_index).
Checks the score of the predicted class (np.squeeze(result)[result_index]):
If the score is below the score_th threshold, the result_index is set to the invalid_value.


Output:
Returns the predicted class index (result_index) if the confidence score exceeds the threshold.
Returns the invalid_value if the confidence score is below the threshold.






SUMMARIES

1. Sign-Language-Interpreter-Reader-and-Transcriber
This program uses the device's camera to capture a person performing sign language.
It transcribes the captured sign language gestures into text and then reads the text aloud using text-to-speech (TTS).



2. KEYPOINT CLASSIFICATION IPYNB
Data Preprocessing: Loads and splits gesture data (keypoints) from a CSV file, where each gesture is represented by 42 features (keypoints from a hand gesture).
Model Creation: Defines and compiles a neural network to classify 26 sign language gestures (A-Z).
Training: Trains the model using the processed data, saves the best model, and stops early if there's no improvement.
Evaluation: Evaluates the model on test data, makes predictions, and converts the trained model into TensorFlow Lite (TFLite) format for use in resource-constrained devices like mobile phones.



3. POINT HISTORY CLASSIFICATION IPYNB
Data Loading and Preprocessing: Loads a sequence of 2D points representing hand gestures and splits them into training and testing datasets.
Model Creation: Defines a neural network (using LSTM or Dense layers) to classify gestures based on the sequence of hand movements.
Training: Trains the model, saves the best model, and performs early stopping.
Evaluation and Conversion: Evaluates the model on test data and converts it to a lightweight TFLite model for easier deployment on mobile devices.



4. APP PY
Camera and Gesture Detection: Captures live video from a camera, detects hand gestures using MediaPipe, and classifies them using the pre-trained models (KeyPointClassifier and PointHistoryClassifier).
Gesture Actions: Maps recognized gestures to actions like typing letters, adding spaces, deleting characters, and reading the word aloud.
Visualization: Displays the recognized gesture on the video feed and updates the word being typed in real-time.



5. KEYPOINT CLASSIFIER PY
Model Initialization: Loads a TFLite model to classify hand gestures based on 2D keypoints.
Inference: Takes a list of keypoints (from the hand landmarks), runs the model, and returns the predicted gesture class.
Output: Returns the predicted gesture label based on the model’s output.



6. POINT HISTORY CLASSIFIER PY
Model Initialization: Loads a TFLite model to classify gestures based on the sequence of hand movement points.
Inference: Takes a sequence of hand movement points, runs inference on the model, and processes the output to return the predicted gesture class.
Confidence Threshold: Ensures that the predicted class has a high enough confidence score before returning the result.







HOW TO RUN PROJECT

ON WINDOWS

python -m venv venv
.\venv\Scripts\activate


ON MAC

python3 -m venv venv
source venv/bin/activate


pip install --upgrade pip     

pip install opencv-python mediapipe tensorflow numpy
pip install pyttsx3


# run code
python app.py
select a python environment if necessary


