import cv2
import os
import subprocess
from PIL import Image,ImageDraw


def detect_face(path,img_file):
    print 'detecting' + path+img_file
    commands = './main -img ' + path+img_file + ' -classifier ./cascade5.xml -minNeighbors 3 -minSize 20 -maxSize 700'
    detect_return = subprocess.check_output(commands, shell=True)
    result = detect_return.split(' ')
    if result[4] == 0:
        print 'no faces detected'
    else:  
        x1 = int(result[0])
        y1 = int(result[1])
        x2 = int(result[2])+ x1
        y2 = int(result[3])+ y1
        img = Image.open(path+img_file)
        draw_instance = ImageDraw.Draw(img)
        draw_instance.rectangle((x1,y1,x2,y2), outline=(255, 0,0))
        img.save('detected'+img_file)
        print 'detection finished'

if __name__ == '__main__':
    VIDEO_PATH = './video/'
    video_files = os.listdir(VIDEO_PATH)
    for each in video_files:
        detect_face(VIDEO_PATH,each)