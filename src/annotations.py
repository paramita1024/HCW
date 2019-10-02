from myutil import *
import numpy as np
import shutil 

label_file = '../Real_Data/EyePAC/labels_part'
path_dest = '../Real_Data/EyePAC/annotation/'
path_src = '../Real_Data/EyePAC/Images_part/'
names = load_data(label_file).keys()[:54]
# for i in range(8):
#     os.mkdir(path_dest + str(i))

for i in range(54):
    for j in range(3):
        key = 3*i + j 
        folder = key % 8
        image_file = path_src + names[i] + '.jpg'
        dest_file = path_dest + str(folder) + '/' + names[i] + '.jpg'
        shutil.copy( image_file, dest_file )
