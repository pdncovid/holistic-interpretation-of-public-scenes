import subprocess
import argparse
from glob import glob
'''
Usage : 
python dataset_labeling_script.py -i ./videos/DEEE -o ./labels/DEEE --suffix yolo -f ../NNHandler_person.py -e mp4
'''


parser = argparse.ArgumentParser()

parser.add_argument("--input", "-i", type=str, dest="input", help="Input folder")
parser.add_argument("--output", "-o", type=str, dest="output", help="Output folder")
parser.add_argument("--suffix", "-s", type=str, dest="suffix", help="Suffix added to json as : vidName_suffix.json")
parser.add_argument("--file", "-f", type=str, dest="file", help="Python file to be run")
parser.add_argument("--ftype", "-e", type=str, dest="ftype", default="mp4", help="Extension of video file")
parser.add_argument("--verbose", "-v", action="store_true", dest="verbose")

args = parser.parse_args()

src = args.input
dest = args.output
suffix = args.suffix
py_file = args.file

src_files = list(map(lambda x:x.replace("\\", "/"), glob(src + "/*.%s"%args.ftype)))
if args.verbose: print("source files :" + "\n\t\t" + "\n\t\t".join(src_files))

for file in src_files:
    file_name = file.split("/")[-1].rstrip(".%s"%args.ftype)
    out_name = dest.rstrip('/') + "/%s-%s.json"%(file_name, suffix)

    cmd = "python %s -i %s -o %s"%(py_file, file, out_name)
    print(cmd)

    subprocess.call(["python", py_file, "-i", file, "-o", out_name])



    # subprocess.call("test1.py", shell=True)