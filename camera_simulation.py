import cv2
import subprocess
import flask
# import flask_cors
import multiprocessing
import argparse

from logging_utils import root_logger
import threading

video_source_app = flask.Flask(__name__)

src = None

# TODO 多视频流同时进入时，需要考虑如何建立多视频流输入机制
# 本地视频流
video_info_list = [
    {"id": 0, "type": "student in classroom", "path": "input/input.mov", "url": "http://127.0.0.1:5912/video"},
    {"id": 1, "type": "people in meeting-room", "path": "input/input1.mp4", "url": "http://127.0.0.1:5912/video"},
    {"id": 3, "type": "traffic flow outdoor", "path": "input/traffic-720p.mp4", "url": "http://127.0.0.1:5912/video"}
]


def get_video_frame():
    assert src
    video_cap = cv2.VideoCapture(src)
    while True:
        ret, frame = video_cap.read()
        if ret:
            ret, jpeg = cv2.imencode('.jpg', frame)
            frame = jpeg.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type:image/jpeg\r\n'
                   b'Content-Length: ' + f"{len(frame)}".encode() + b'\r\n'
                                                                    b'\r\n' + frame + b'\r\n')
        else:
            video_cap = cv2.VideoCapture(src)


@video_source_app.route('/video')
# @flask_cors.cross_origin()
def read_video():
    return flask.Response(get_video_frame(),
                          mimetype='multipart/x-mixed-replace; boundary=frame')


def start_video_stream(video_src, port):
    global src
    src = video_src
    video_source_app.run(host="0.0.0.0", port=port)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', dest='video',
                        type=int, default=0)
    parser.add_argument('--port', dest='port',
                        type=int, default=5912)

    args = parser.parse_args()

    assert args.video in [x['id'] for x in video_info_list]
    video_info = video_info_list[args.video]
    root_logger.info('start video stream of ')

    start_video_stream(video_info['path'], args.port)



