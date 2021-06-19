
from NNHandler_image import NNHandler_image, cv2

in_name = "./data/videos/EarthCam/long/Dublin Cam.ts"
out_prefix = "./data/videos/EarthCam/dublin_"

# vid = NNHandler_image(format="ts", img_loc="./data/videos/EarthCam/long/Dublin Cam.ts")
vid = NNHandler_image(format="ts", img_loc=in_name)

vid.open(init_param=True)
print(vid)

# vid.cap.set(cv2.CAP_PROP_FPS, 30)
# print(vid.count_frames())
# vid.init_param()
# print(vid)

ind = 0
i = 0
out = NNHandler_image(format="avi")
out.init_writer(out_name=out_prefix + "{}.avi".format(ind), h=vid.height, w=vid.width)

while True:

    frame = vid.read_frame()
    if frame is None:
        out.close_writer()
        break
    else:
        out.write_frame(frame)

    if (i != 0 and i%1800 == 0):
        out.close_writer()
        print(out_prefix + "{}.avi".format(ind))
        ind += 1

        out.init_writer(out_name=out_prefix + "{}.avi".format(ind), h=vid.height, w=vid.width)

    i += 1


vid.close()
