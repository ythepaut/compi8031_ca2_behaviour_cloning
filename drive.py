import socketio
import eventlet.wsgi
from flask import Flask
from keras.models import load_model
import base64
from io import BytesIO
from PIL import Image
import numpy as np

from behaviouralCloning import preprocess_img

SPEED_LIMIT = 10

sio = socketio.Server()
app = Flask(__name__)


def send_control(steering_angle, throttle):
    sio.emit("steer", data={
        "steering_angle": steering_angle.__str__(),
        "throttle": throttle.__str__()
    })


@sio.on("connect")
def connect(sid, env):
    print('Connected')
    send_control(0, 0)


@sio.on("telemetry")
def telemetry(sid, data):
    try:
        image = Image.open(BytesIO(base64.b64decode(data["image"])))
        image = np.asarray(image)
        image = preprocess_img(image, True)
        image = np.array([image])
        speed = float(data['speed'])
        throttle = 1.0 - speed / SPEED_LIMIT
        steering_angle = float(model.predict(image))
        send_control(steering_angle, 1)
    except TypeError:
        pass


if __name__ == "__main__":
    model = load_model("model.h5")
    app = socketio.Middleware(sio, app)
    eventlet.wsgi.server(eventlet.listen(("", 4567)), app)
