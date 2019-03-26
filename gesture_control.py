#import os
#import math
#import random
import numpy as np
import tensorflow as tf
import cv2
import keyboard
import time
from pynput.mouse import Button, Controller
import wx
import playsound
slim = tf.contrib.slim
#import matplotlib.pyplot as plt
#import matplotlib.image as mpimg
import sys
sys.path.append('../')
from nets import ssd_vgg_300, ssd_common, np_methods
from preprocessing import ssd_vgg_preprocessing
from notebooks import visualization

app = wx.App(False)
mouse = Controller()
screen_x, screen_y = wx.GetDisplaySize()

# TensorFlow session: grow memory when needed. TF, DO NOT USE ALL MY GPU MEMORY!!!
gpu_options = tf.GPUOptions(allow_growth=True)
config = tf.ConfigProto(log_device_placement=False, gpu_options=gpu_options)
isess = tf.InteractiveSession(config=config)

# Input placeholder.
net_shape = (300, 300)
data_format = 'NHWC'
img_input = tf.placeholder(tf.uint8, shape=(None, None, 3))

# Evaluation pre-processing: resize to SSD net shape.
image_pre, labels_pre, bboxes_pre, bbox_img = ssd_vgg_preprocessing.preprocess_for_eval(
    img_input, None, None, net_shape, data_format, resize=ssd_vgg_preprocessing.Resize.WARP_RESIZE)
image_4d = tf.expand_dims(image_pre, 0)

# Define the SSD model.
reuse = True if 'ssd_net' in locals() else None
ssd_net = ssd_vgg_300.SSDNet()
with slim.arg_scope(ssd_net.arg_scope(data_format=data_format)):
    predictions, localisations, _, _ = ssd_net.net(image_4d, is_training=False, reuse=reuse)

# Replace the path with the model file given in description
ckpt_filename = r'C:\Users\PRAVEEN\Desktop\Capstone_ssd\SSD-Tensorflow-master\train_model_fine_tune_3\model.ckpt-2870'
isess.run(tf.global_variables_initializer())
saver = tf.train.Saver()
saver.restore(isess, ckpt_filename)
# SSD default anchor boxes.
ssd_anchors = ssd_net.anchors(net_shape)

# Main image processing routine.
def process_image(img, select_threshold=0.3, nms_threshold=.8, net_shape=(300, 300)):
    # Run SSD network.
    rimg, rpredictions, rlocalisations, rbbox_img = isess.run([image_4d, predictions, localisations, bbox_img],
                                                              feed_dict={img_input: img})

    # Get classes and bboxes from the net outputs.
    rclasses, rscores, rbboxes = np_methods.ssd_bboxes_select(
        rpredictions, rlocalisations, ssd_anchors,
        select_threshold=select_threshold, img_shape=net_shape, num_classes=5, decode=True)

    rbboxes = np_methods.bboxes_clip(rbbox_img, rbboxes)
    rclasses, rscores, rbboxes = np_methods.bboxes_sort(rclasses, rscores, rbboxes, top_k=400)
    rclasses, rscores, rbboxes = np_methods.bboxes_nms(rclasses, rscores, rbboxes, nms_threshold=nms_threshold)
    # Resize bboxes to original image shape. Note: useless for Resize.WARP!
    rbboxes = np_methods.bboxes_resize(rbbox_img, rbboxes)
    return rclasses, rscores, rbboxes
	
	
	
cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_SIMPLEX

#initialising defaults
pixel_threshold_1 = 30
pixel_threshold_2 = 40
pixel_threshold_3 = 40
thresh_val = 10

#count_l_dir = 0
start_pauser = 0
start_appch = 0
start_mouse = 0
start_l_mouse = 0
start_lc = 0

count_dir = 0
count_appch = 0
count_l_appch = 0
count_m = 0
#count_lc = 0

click_flag = 1
mouse_active = 1
# Define the codec and create VideoWriter object
#fourcc = cv2.VideoWriter_fourcc(*'XVID')
#out = cv2.VideoWriter('output.avi',fourcc, 20.0, (640,480))

while(True):
    #print(rclasses)
    # Capture frame-by-frame
    ret, img = cap.read()
    
    #if ret==True:
        #frame = cv2.flip(img,1)

        # write the flipped frame
        #out.write(frame)
    
    #info on height and width of each frame
    height = img.shape[0]
    width = img.shape[1]
    # Our operations on the frame come here
    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rclasses, rscores, rbboxes =  process_image(img)
    
    if time.time() - start_pauser > 2:
        count_dir = 0
        count_l_dir = 0
    
    if time.time() - start_appch > 1.5:
        if count_appch > 0:
            keyboard.release('alt')
        count_appch = 0
        count_l_appch = 0
        
        
    if time.time() - start_l_mouse > 2:
        count_m = 0

        
    
    if rclasses.any() > 0:
        #print(rclasses)
        ymin = int(rbboxes[0, 0] * height)
        xmin = int(rbboxes[0, 1] * width)
        ymax = int(rbboxes[0, 2] * height)
        xmax = int(rbboxes[0, 3] * width)
        # Display the resulting frame
        cv2.putText(img, (str(rclasses[0]) + "-" + str(rscores[0])),(xmax, ymax), font, 2, (26,255,255))
        #cv2.putText(img, str(rscores[0]),(int((xmax+xmin)/2), ymax), font, 2, (255,255,255))
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        cv2.imshow('frame',img)
        count_dir += 1
        count_appch += 1
        count_m += 1
        #dynamic thresholding
        area = ((xmin - xmax)*(ymin - ymax))/100
        pixel_threshold_1 = pixel_threshold_2 = pixel_threshold_3 = area/thresh_val
        #print(pixel_threshold_1)
        #keyboard controls for up, down, left, right, space
        if (rclasses[0] in [4,5]) and mouse_active > 0:
            x_mid = (xmin + xmax)/2
            y_mid = (ymin + ymax)/2
            if count_dir == 1:
                start_pauser = time.time()
                x_mid_1 = x_mid
                y_mid_1 = y_mid
            elif ((count_dir > 1) and ((time.time() - start_pauser) > 0.5) and (x_mid < (x_mid_1 + pixel_threshold_1)) and (x_mid > (x_mid_1 - pixel_threshold_1)) and (y_mid < (y_mid_1 + pixel_threshold_1)) and (y_mid > (y_mid_1 - pixel_threshold_1))):
                keyboard.press_and_release('space')
                #print("space")
                time.sleep(0.75)
                #count = 0
            elif ((count_dir > 1) and (x_mid > (x_mid_1 + pixel_threshold_2)) and ((time.time() - start_pauser) > 0.15)):
                keyboard.press_and_release('left arrow')
                #print("left arrow")
                #if count_l > 3:
                #    keyboard.press('left arrow')
                #time.sleep(0.25)
                #count_l += 1
                start_pauser = time.time()
            elif ((count_dir > 1) and (x_mid < (x_mid_1 - pixel_threshold_2)) and ((time.time() - start_pauser) > 0.15)):
                keyboard.press_and_release('right arrow')
                #print("right arrow")
                #time.sleep(0.75)
                #count_l += 1
                start_pauser = time.time()
            elif ((count_dir > 1) and (y_mid > (y_mid_1 + pixel_threshold_3)) and ((time.time() - start_pauser) > 0.15)):
                #print('yyyyyyy')
                keyboard.press_and_release('down arrow')
                #time.sleep(0.75)
                #count_l += 1
                start_pauser = time.time()
            elif ((count_dir > 1) and (y_mid < (y_mid_1 - pixel_threshold_3)) and ((time.time() - start_pauser) > 0.15)):
                #print('nnnnnnnnnn')
                keyboard.press_and_release('up arrow')
                #time.sleep(0.75)
                #count_l += 1
                start_pauser = time.time()
                
        #task switcher function        
        if (rclasses[0] in [3]) and mouse_active > 0:
            x_mid = (xmin + xmax)/2
            y_mid = (ymin + ymax)/2
            if count_appch == 1:
                start_appch = time.time()
            elif count_appch > 1 and (time.time() - start_appch) > 0.5 and count_l_appch == 0:
                x_mid_1 = x_mid
                y_mid_1 = y_mid
                keyboard.press('alt')
                count_l_appch += 1
            elif ((count_appch > 1) and (count_l_appch > 0) and ((time.time() - start_appch) > 0.15) and (x_mid < (x_mid_1 + pixel_threshold_1)) and (x_mid > (x_mid_1 - pixel_threshold_1)) and (y_mid < (y_mid_1 + pixel_threshold_1)) and (y_mid > (y_mid_1 - pixel_threshold_1))):
                keyboard.press_and_release('tab')
                time.sleep(0.75)
                start_appch = time.time()
        
        #switch between mouse and keyboard
        if rclasses[0] in [1]:
            if count_m == 1:
                start_mouse = time.time()
                start_l_mouse = time.time()
            elif count_m > 1:
                start_l_mouse = time.time()
            if (time.time() - start_mouse) > 2:
                mouse_active = mouse_active * (-1)
                if mouse_active < 0:
                    playsound.playsound('mouse activated.mp3')
                elif mouse_active > 0:
                    playsound.playsound('mouse deactivated.mp3')
                time.sleep(2)
                
        #mouse functions
        if mouse_active < 0:
            if rclasses[0] in [4]:
                #x_mid = (xmin + xmax)/2
                #y_mid = (ymin + ymax)/2
                #mouse.position = (screen_x -  (xmin * screen_x/(width - 150)), ymin * screen_y/(height - 200))
                x_mid = (width/2 - (xmin + xmax)/2)/10
                y_mid = ((ymin + ymax)/2 - height/2)/10
                #if x_mid > 0 :
                #    x_mid = 5
                #else:
                #    x_mid = -5
                #if y_mid > 0:
                #    y_mid = 5
                #else:
                #    y_mid = -5
                mouse.move(x_mid, y_mid)
                
            if rclasses[0] in [2]:
                if click_flag > 0:
                    mouse.press(Button.left)
                    click_flag = click_flag * (-1)
                start_lc = time.time()
                x_mid = (width/2 - (xmin + xmax)/2)/10
                y_mid = ((ymin + ymax)/2 - height/2)/10
                mouse.move(x_mid, y_mid)
                #count_lc += 1
            if (time.time() - start_lc) > 0.25 and click_flag < 0:
                mouse.release(Button.left)
                click_flag = 1
                #count_lc = 0
                
            if rclasses[0] in [5]:
                mouse.click(Button.left, 2)
                time.sleep(0.5)
     
        #if rclasses[0] in [1,2,3]:    
    
    cv2.imshow('frame',img)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
#out.release()
cv2.destroyAllWindows()	
	
