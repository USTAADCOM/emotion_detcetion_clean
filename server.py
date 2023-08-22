"""
Emotion detection application flask server main page.
"""
from flask import Flask, render_template, Response
from model_detection import VideoCamera

app = Flask(__name__)

def gen_frames(videocamera_object):
    """
    method yiled frame from get_frame() method 
    after emotion detection.

    Parameters
    ----------
    None

    Return
    ------
    frame: yiled
        yield live streaming frame with emotion detection and 
        landmarks.
    """
    while True:
        frame = videocamera_object.get_frame()
        yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  
@app.route('/')
def index():
    """
    Index page emotion detcetion vide streaming application.
    
    Parameters
    ----------
    None

    Return
    ------
        render template index.html.
    """
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """
    method return response of yiled frame from gen_frames() method 
    after emotion detection.

    Parameters
    ----------
    None

    Return
    ------
    frame: Response
        return live streaming frame with emotion detection and 
        landmarks.
    """
    return Response(gen_frames(VideoCamera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)




# while True:
#         ret,frame = video.read()
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         faces = faceDetect.detectMultiScale(gray, 1.3, 3)
#         rects = detector(gray, 0)
#         for x,y,w,h in faces:
#             sub_face_img = gray[y:y+h, x:x+w]
#             resized=cv2.resize(sub_face_img,(48,48))
#             normalize=resized/255.0
#             reshaped=np.reshape(normalize, (1, 48, 48, 1))
#             result = model.predict(reshaped)
#             label = np.argmax(result, axis=1)[0]
#             for (i, rect) in enumerate(rects):
#             # determine the facial landmarks for the face region, then
#             # convert the facial landmark (x, y)-coordinates to a NumPy
#             # array
#                 shape = predictor(gray, rect)
#                 shape = face_utils.shape_to_np(shape)
#             # loop over the (x, y)-coordinates for the facial landmarks
#             # and draw them on the image
#             for (x, y) in shape:
#                 cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)
#             print(label)
#             cv2.rectangle(frame, (x,y), (x+w, y+h), (0,0,255), 1)
#             cv2.rectangle(frame,(x,y),(x+w,y+h),(50,50,255),2)
#             cv2.rectangle(frame,(x,y-40),(x+w,y),(50,50,255),-1)
#             cv2.putText(frame, labels_dict[label], (x, y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)