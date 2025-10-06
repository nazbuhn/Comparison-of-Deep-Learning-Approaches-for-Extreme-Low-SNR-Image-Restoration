import os
import sys

name = sys.argv[1]
in_dir = '/data2/hilo2-512/' + name + '/raw'
out_dir = '/data2/hilo2-512-noise2fast/' + name

os.makedirs(out_dir,exist_ok=True)

print('running' + in_dir)

while len(os.listdir(in_dir)) != len(os.listdir(out_dir)):
    cmd = 'python N2F.py ' + in_dir + ' ' + out_dir
    os.system(cmd)
