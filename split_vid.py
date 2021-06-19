
from NNHandler_image import NNHandler_image, cv2

vid = NNHandler_image(format="ts", img_loc="./data/videos/EarthCam/long/Dublin Cam.ts")
# vid = NNHandler_image(format="ts", img_loc="../data/videos/EarthCam/Dublin Cam.ts")

vid.open(init_param=True)
print(vid)

# vid.cap.set(cv2.CAP_PROP_FPS, 30)
# print(vid.count_frames())
# vid.init_param()
# print(vid)

ind = 0
i = 0
out = NNHandler_image(format="mp4")
out.init_writer(out_name="./data/videos/EarthCam/dublin_{}.mp4".format(ind), h=vid.height, w=vid.width)

while True:
    frame = vid.read_frame()
    if frame is None:
        out.close_writer()
        break
    else:
        out.write_frame(frame)

    if (i != 0 and i%1800 == 0):
        out.close_writer()
        print("../data/videos/EarthCam/dublin_{}.mp4".format(ind))
        ind += 1


        out.init_writer(out_name="../data/videos/EarthCam/dublin_{}.mp4".format(ind), h=vid.height, w=vid.width)

    i += 1


vid.close()
