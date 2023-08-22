# Real Time Emotion Detcetion
Detcet and predict the facial emotion from live stream.

## Setup
  
  clone project with Python 3.10.10
  ```code
  git clone https://github.com/USTAADCOM/emotion_detcetion.git
  cd emotion_detcetion
  pip install -r requirements.txt
  ```
## Model
  Download Model Link below

  [Models](https://drive.google.com/file/d/1_veBZVgjXQaXTIHTtq-JKJNrCpJsEQBG/view?usp=sharing) 

  Extract and copy model directory in emotion_detection folder
## Project Structure

```bash
emotion_detection_clean
  │   model_detection.py
  │   README.md
  │   requirements.txt
  │   server.py
  │
  ├───models
  │       haarcascade_frontalface_default.
  │       Keras_emotion_detection_model_tr
  │       model_emotion.h5
  │       shape_predictor_68_face_landmark
  │
  ├───templates
  │       index.html
  │
  ├───utils
  │   data_land_marker.py
  │     image_classifier.py
```

## Real Time Facial Emotion Detection 
```code
python server.py
```
