from suren.util import Json

class YOLO_Handshake():
    VARS = {
        "n_frame" : "noFrames",
        "n_box" : "noBboxes"
    }

    def __init__(self, loc='yoloHandshakeOutput.json'):
        js = Json(loc)

        self.yolo_data = js.read()

        self.TIME_SERIES_LENGTH = self.yolo_data["noFrames"]


