"""
Main module for feature extraction and prediction for emotion detction
"""
import cv2
import numpy as np
from keras.models import load_model
from utils.data_land_marker import LandMarker
from utils.image_classifier import ImageClassifier

# Load Models
model = load_model('models/model_emotion.h5')
PREDICTOR_PATH = 'models/shape_predictor_68_face_landmarks.dat'
faceDetect = cv2.CascadeClassifier('models/haarcascade_frontalface_default.xml')

land_marker = LandMarker(landmark_predictor_path = PREDICTOR_PATH)
classifier = ImageClassifier(land_marker = land_marker)

WHITE_COLOR = (255, 255, 255)
GREEN_COLOR = (0, 255, 0)
BLUE_COLOR = (255, 255, 104)
INDEX_NUM = 0
default_cam = [-1, 0, 1]
for INDEX_NUM in default_cam:
    frame_temp = cv2.VideoCapture(INDEX_NUM)
    if frame_temp.isOpened():
        break
video = cv2.VideoCapture(INDEX_NUM)
# video = cv2.VideoCapture(0)
# video = cv2.VideoCapture("video.mp4")
labels_dict = {0:'Angry',1:'Disgust', 2:'Fear', 3:'Happy',4:'Neutral',5:'Sad',6:'Surprise'}
landmark_points_list = [[(0, 0)]]

def draw_landmark_points(points: np.ndarray, img, color = BLUE_COLOR):
    """
    method take the 68 points array and draw over the image
    with specifief color.

    Parameters
    ----------
    points: array
        array of 68 points represeting face features Nose, Mouth etc.
    img: Image
        origional frame image 
    color: Tuple
        color of the face landmark.

    Return
    ------
    None 
        if no points provided for face over the image.
    """
    if points is None:
        return None
    for (x_axis, y_axis) in points:
        cv2.circle(img, (x_axis, y_axis), 1, color, -1)

class VideoCamera(object):
    """
    Class Doc string.
    """
    def __init__(self):
        self.video = cv2.VideoCapture(INDEX_NUM)
        # self.video = cv2.VideoCapture("video.mp4")

    def __del__(self):
        self.video.release()

    def get_frame(self):
        """
        method yiled frame from get_frame() method 
        after emotion detection.

        Parameters
        ----------
        None

        Return
        ------
        frame: bytes 
            return live streaming frame with emotion detection and 
            landmarks in bytes form.
        """
        _,frame = video.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = faceDetect.detectMultiScale(gray, 1.3, 3)
        landmark_points_list = classifier.extract_landmark_points(img = frame)
        for x_axis, y_axis, width, height in faces:
            sub_face_img = gray[y_axis : y_axis + height, x_axis : x_axis + width]
            resized = cv2.resize(sub_face_img,(48,48))
            normalize = resized/255.0
            reshaped = np.reshape(normalize, (1, 48, 48, 1))
            result = model.predict(reshaped)
            label = np.argmax(result, axis=1)[0]
            for lm_points in landmark_points_list:
                draw_landmark_points(points = lm_points, img = frame)
            print(label)
            cv2.rectangle(frame, (x_axis, y_axis),
                          (x_axis + width, y_axis + height), (0,0,255), 1)
            cv2.rectangle(frame, (x_axis, y_axis),
                          (x_axis + width, y_axis + height),(50,50,255),2)
            cv2.rectangle(frame, (x_axis, y_axis - 40),
                          (x_axis + width, y_axis),(50,50,255),-1)
            cv2.putText(frame, labels_dict[label], (x_axis, y_axis - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
        _, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes()
