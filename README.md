# Real-Time Sign Language Recognition and Translation System
## Overview:
A real-time sign language recognition system that captures video through a webcam, extracts hand pose keypoints using MediaPipe, and uses a deep learning model to classify the performed sign. The system then translates these recognized signs into natural language sentences.

![image](https://github.com/user-attachments/assets/dd16ee86-e7bc-4b04-bc7c-dd1638814ea2)

# Key Components:
### Data Collection & Preprocessing
The input videos of sign language gestures are manually curated and processed into 60-frame sequences. Each frame originally contains 1662 landmark values, but only the hand keypoints are used for training—specifically, 21 landmarks per hand (totaling 63 values per hand), along with 2 presence flags. To handle missing hand data, neutral poses are generated using clustering techniques and used to mask the gaps. This ensures consistent input dimensions and reduces noise during training.


## Model Architecture

- A **Temporal Convolutional Network (TCN)** and **LSTM variant** have been tested.  
- **Input shape:** (60 frames, 128 features)  
- **Output:** One of the predefined sign classes (excluding "no gesture")  
- **Accuracy:** Over **80% validation accuracy**, depending on the model.

  
## LSTM Architecture:

![image](https://github.com/user-attachments/assets/dfa6de43-aab4-4b0a-b5f9-72f200272e98)
## Model Program
```py
def build_lstm_model(input_shape, num_classes):
    model = Sequential([
        Input(shape=input_shape),
        Dropout(0.2),
        LSTM(64, return_sequences=True),
        Dropout(0.3),
        LSTM(64),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer=Adam(learning_rate=0.0005),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

input_shape = (30, 128) 
num_classes = len(class_names)
model = build_lstm_model(input_shape, num_classes)
```


## Real-Time Inference
For real-time sign recognition, MediaPipe is used to extract keypoints from a live video feed. The system maintains a rolling window of 60 frames and continuously predicts signs based on this sequence. To avoid false positives, it excludes predictions of "no gesture" when idle. The list of valid sign classes is dynamically loaded from the dataset folder structure. Once predictions are made, the system also includes a Natural Language Translation component that converts raw sign sequences like "I GO SCHOOL" into grammatically correct English, such as "I am going to school", adapting tense and phrasing based on context.
 
  



# Demo video:

https://github.com/user-attachments/assets/e65b60b5-0f8f-4ded-a5ec-21b88582756a


# Output Graph:
![image](https://github.com/user-attachments/assets/ea9c5429-6fae-4e32-8ba4-9bfbc2c365bd)

## Deployment Goal
### Build a Web-Based Interface:

- **Frontend:** Captures video input and sends extracted keypoints to the backend  
- **Backend:** Runs the trained model and returns recognized signs  
- **Display:** Shows **live predictions** and **translated sentences** on the screen

## Conclusion
The project successfully achieves real-time sign language recognition with over 82% validation accuracy, using a manually trained dataset of 100+ actions. By combining MediaPipe keypoint extraction, neutral pose masking, and LSTM-based modeling, the system delivers accurate predictions and translates gesture sequences into natural language. It’s optimized for live deployment, paving the way for inclusive and accessible communication.

