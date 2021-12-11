import argparse
import json

if __name__=="__main__":
    args=argparse.ArgumentParser()
    print("a")
    args.add_argument("--predJSON", type=str, default="../data/labels/UTI/seq18/mask.json" ,help="ground truth json file")
    args.add_argument("--groundTruthFolder" , type=str,default="../data/ground_truth/UTI/ut-interaction_set2/seq18-mask_GT",help="output folder")
    args.add_argument("--tempFolder", type=str, default="./temp")
    args=args.parse_args()

    predDict=json.load(open(args.predJSON))
    noFramesPred=predDict["frames"]

    for n in range(noFramesPred):
        frame=predDict[str(n)]
        st=""
        for face in frame:
            st=st+"face 1.00 {} {} {} {}\n".format(int(face["x1"]),int(face["y1"]),int(face["x2"]),int(face["y2"]))
        with open(args.tempFolder+"/detections/frame{}.txt".format(n),"w+") as f:
            f.write(st)
        print("Frame GT {} done".format(n))


        with open(args.groundTruthFolder+"/{}.txt".format(n),"r") as f:
            lines=f.readlines()
            st=""
            for l in lines:
                ar = l.strip().split()
                st=st+"face {} {} {} {}\n".format(int(float(ar[1])),int(float(ar[2])),\
                    int(float(ar[3])),int(float(ar[4])))

            with open(args.tempFolder+"/groundtruths/frame{}.txt".format(n),"w+") as f:
                f.write(st)        
