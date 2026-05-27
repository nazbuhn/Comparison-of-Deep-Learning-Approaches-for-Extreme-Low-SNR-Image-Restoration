import os
import sys

in_dir = sys.argv[1]
out_dir = sys.argv[2]

os.makedirs(out_dir,exist_ok=True)

print('running' + in_dir)

while len(os.listdir(in_dir)) != len(os.listdir(out_dir)):
    cmd = 'python N2F.py ' + in_dir + ' ' + out_dir
    os.system(cmd)
