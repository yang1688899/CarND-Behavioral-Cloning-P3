# -*- coding: utf-8 -*-
"""
Created on Tue Oct  3 18:14:32 2017

@author: yang
"""
import csv
import random
import cv2
import matplotlib.pyplot as plt
#get all image path and steering angle of a given diractory
def get_data(dir_name):
    file_path = '../%s/driving_log.csv'%dir_name
    img_paths = []
    steerings = []
    with open(file_path) as csvFile:
        reader = csv.DictReader(csvFile)
        for line in reader:
            img_paths = img_paths+[line['center'],line['left'],line['right']]
            steerings = steerings + [float(line['steering']),float(line['steering'])+0.2,float(line['steering'])-0.2]
        img_paths = [get_full_path(dir_name,path) for path in img_paths]
    return img_paths,steerings

#get the path of a data image in the given diractory
def get_full_path(dir_name,path):
    path = path.replace('/','\\')
    fileName = path.split('\\')[-1]
    full_path = '../%s/IMG/%s'%(dir_name,fileName)
    return full_path

#get rip part of the low steering angle data
def filter_data(img_paths,steerings):
    img_paths_filtered = []
    steerings_filtered = []
    for i in range(len(steerings)):
        if float(steerings[i])>0.0 and float(steerings[i])<0.1:
            if round(random.random()*0.58):
                img_paths_filtered.append(img_paths[i])
                steerings_filtered.append(steerings[i])
                    
        elif float(steerings[i])>-0.2 and float(steerings[i])<0.2:
            if round(random.random()):
                img_paths_filtered.append(img_paths[i])
                steerings_filtered.append(steerings[i])
        else:
            img_paths_filtered.append(img_paths[i])
            steerings_filtered.append(steerings[i])
    return img_paths_filtered,steerings_filtered
                    
                    

img_paths, steerings = get_data('data_second_track')
print (len(img_paths), len(steerings))
img_paths_filtered, steerings_filtered = filter_data(img_paths,steerings) 
print (len(img_paths_filtered), len(steerings_filtered))

plt.imshow(cv2.imread(img_paths[1])) 
  