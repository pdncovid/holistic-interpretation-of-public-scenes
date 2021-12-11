import argparse
import json

if __name__=="__main__":
    args=argparse.ArgumentParser()
    print("a")
    args.add_argument("--predJSON", type=str, default="../data/labels/UTI/seq18/mask.json" ,help="ground truth json file")
    args.add_argument("--groundTruthFolder" , type=str,default="../data/ground_truth/UTI/ut-interaction_set2/seq18-mask_GT",help="output folder")
    args.add_argument("--tempFolder", type=str, default="./temp")
    args.add_argument("--testType", type=str, default="maskAgnostic", help="maskAgnostic or maskAware")
    args=args.parse_args()

    predDict=json.load(open(args.predJSON))
    noFramesPred=predDict["frames"]
    outFileIndex=0
    for n in range(noFramesPred):
        try:
            frame=predDict[str(n)]
            s=""
            for face in frame:
                if args.testType=="maskAgnostic":
                    s=s+"face 0.99 {} {} {} {}\n".format(int(face["x1"]),int(face["y1"]),int(face["x2"]),int(face["y2"]))
                else:
                    s=s+"{} 0.99 {} {} {} {}\n".format(face["name"],int(face["x1"]),int(face["y1"]),int(face["x2"]),int(face["y2"]))
            print("Frame GT {} done".format(n))


            with open(args.groundTruthFolder+"/{}.txt".format(n),"r") as f:
                lines=f.readlines()
                st=""
                for l in lines:
                    ar = l.strip().split()
                    if args.testType=="maskAgnostic":
                        st=st+"face {} {} {} {}\n".format(int(float(ar[1])),int(float(ar[2])),\
                            int(float(ar[3])),int(float(ar[4])))
                    else:
                        st=st+"{} {} {} {} {}\n".format(str(ar[0]).strip(),int(float(ar[1])),int(float(ar[2])),\
                            int(float(ar[3])),int(float(ar[4])))
                
                
                if len(st.strip())>0 and len(s.strip())>0:
                    
                    with open(args.tempFolder+"/detections/frame_{}.txt".format(outFileIndex),"w+") as f:
                        f.write(s.strip())
                    with open(args.tempFolder+"/groundtruths/frame_{}.txt".format(outFileIndex),"w+") as f:
                        f.write(st.strip())
                    outFileIndex+=1

                    if outFileIndex>73:
                        break

        except:
            print("Skipping fame {}".format(n))
    print("END of Python code")
