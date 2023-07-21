import cv2
import subprocess
import flask
import flask_cors

video_source_app = flask.Flask(__name__)

src = "input/input.mov"


def get_video_frame():
    video_cap = cv2.VideoCapture(src)
    while True:
        ret, frame = video_cap.read()
        if ret:
            ret, jpeg = cv2.imencode('.jpg', frame)
            frame = jpeg.tobytes()
            yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n'
                   + frame + b'\r\n\r\n')
        else:
            video_cap = cv2.VideoCapture(src)


@video_source_app.route('/video')
@flask_cors.cross_origin()
def read_video():
    return flask.Response(get_video_frame(),
                          mimetype='multipart/x-mixed-replace; boundary=frame')


def start_video_stream():
    video_source_app.run(host="0.0.0.0", port=5912)
