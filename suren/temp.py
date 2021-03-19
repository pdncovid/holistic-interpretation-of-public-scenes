from suren.util import  Json
import json

# with open('./nn-outputs/yoloOut.txt') as file:
#     data = file.read()
#     data = json.dump

js = Json('../nn-outputs/yoloExp.txt')

js_ = Json('../nn-outputs/sample-YOLO-bbox.json')

js_.write(js.read())
# print(js.read())