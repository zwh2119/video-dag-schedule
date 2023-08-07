import cv2
import subprocess
import flask
# import flask_cors
import multiprocessing
import argparse
import os

import yaml_utils

from logging_utils import root_logger
import threading

video_source_app = flask.Flask(__name__)


# TODO 多视频流同时进入时，需要考虑如何建立多视频流输入机制
# 本地视频流


configs = yaml_utils.read_yaml('configure.yaml')
video_info_list = configs['video']
src = ''

def get_video_frame():
    global src
    assert src, 'None video path found!'
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
def read_video():
    return flask.Response(get_video_frame(),
                          mimetype='multipart/x-mixed-replace; boundary=frame')


def start_video_stream(video_src, port):
    assert os.path.exists(video_src), "video path does not exist!"
    global src
    src = video_src
    video_source_app.run(host="0.0.0.0", port=port)


if __name__ == '__main__':

    for video in video_info_list:
        if video['state']:
            root_logger.info(f'start video stream of video {video["id"]}')
            multiprocessing.Process(target=start_video_stream, args=(video['path'], video['port'])).start()



