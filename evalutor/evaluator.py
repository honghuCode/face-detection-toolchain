'''
Face Detection Evaluator
'''
import cv2
import os
import sys
import argparse
import subprocess
import numpy as np 
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

class ProgressBar:
    def __init__(self, count = 0, total = 0, width = 50):
        self.count = count
        self.total = total
        self.width = width
    def move(self):
        self.count += 1
        self.log()
    def log(self):
        sys.stdout.write(' ' * (self.width + 9) + '\r')
        sys.stdout.flush()
        progress = self.width * self.count / self.total
        sys.stdout.write('{0:3}/{1:3}: '.format(self.count, self.total))
        sys.stdout.write('=' * progress + '-' * (self.width - progress) + '\r')
        if progress == self.width:
            sys.stdout.write('\n')
        sys.stdout.flush()

progressbar = ProgressBar(total=13500)
mblbpcascade='./classifier/mblbp-2.xml'

def load_data():
    NEG_PATH = './data/neg/'
    POS_PATH = './data/pos/'
    neg_files = os.listdir(NEG_PATH)
    pos_files = os.listdir(POS_PATH)
    return POS_PATH, NEG_PATH, pos_files, neg_files

def _mblbp_face_detect(img_file,threshold,cascadename='./cascade2.xml'):
    commands = './mblbpdetect -img ' + img_file + ' -classifier ' + cascadename + ' -minNeighbors ' + str(threshold) + ' -minSize 20' + ' -maxSize 700'
    result = subprocess.check_output(commands, shell=True)
    face_num = int(result)
    return face_num

def _ocv_face_detect(img_file, feature_type, threshold):
    img = cv2.imread(img_file)
    cascadefile = None
    gray = None
    if feature_type == 'haar':
        cascadefile = './classifier/haar.xml'
    elif feature_type == 'lbp':
        cascadefile = './classifier/lbp.xml'
    elif feature_type == 'mblbp':
        raise ValueError('MBLBP is Not Supported by OpenCV')
    else:
        raise ValueError('Cascade Not Supported')
    face_cascade = cv2.CascadeClassifier(cascadefile)
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2 ,minNeighbors=threshold)
    result = []
    for (x,y,width,height) in faces:
        result.append((x,y,x+width,y+height))
    return result

def _haar_face_detect(img_file,threshold):
    return len(_ocv_face_detect(img_file,'haar',threshold))

def _lbp_face_detect(img_file,threshold):
    return len(_ocv_face_detect(img_file,'lbp',threshold))

def get_face_num_from_img(img_file, feature_type, threshold,mblbpcascade):
    if feature_type =='haar':
        return _haar_face_detect(img_file, threshold)
    elif feature_type =='lbp':
        return _lbp_face_detect(img_file, threshold)
    elif feature_type == 'mblbp':
        return _mblbp_face_detect(img_file, threshold,cascadename = mblbpcascade)
    else:
        raise ValueError('Feature Type Not Supported')

def single_threshold_test(feature_type,threshold,mblbpcascade):
    tp = 0
    fn = 0
    tn = 0
    fp = 0
    POS_PATH, NEG_PATH, pos_files, neg_files = load_data()
    for each_file in pos_files:
        face_num = int(get_face_num_from_img(POS_PATH+each_file,feature_type,threshold,mblbpcascade))
        progressbar.move()
        if (face_num >= 1):
            tp = tp + 1
        else:
            fn = fn + 1
    for each_file in neg_files:
        face_num = int(get_face_num_from_img(NEG_PATH+each_file,feature_type,threshold,mblbpcascade))
        progressbar.move()
        if (face_num == 0):
            tn = tn + 1
        else:
            fp = fp +1
    # Check if all pictures have been checked
    if ( (tp+fn) != len(pos_files) or (tn+fp)!=len(neg_files) ):
        raise ValueError('Not all samples have been checked')
    print 'Selected Feature Type :'+ feature_type
    print 'Current Threshold :' + str(threshold)
    print 'True Positive :'+ str(tp)
    print 'True Negative :'+ str(tn)
    print 'False Positive :'+ str(fp)
    print 'False Negative :'+ str(fn)
    return tp, tn, fp, fn

def multiple_threshold_test(feature_type, thresholds,mblbpcascde):
    tps =[]
    tns =[]
    fps =[] 
    fns =[]
    for each_threshold in thresholds:
        tp, tn, fp, fn = single_threshold_test(feature_type,each_threshold,mblbpcascde)
        tps.append(tp)
        tns.append(tn)
        fps.append(fp)
        fns.append(fn)
    return tps, tns, fps, fns


def all_test(mblbpcascade='cascade2.xml'):
    progressbar = ProgressBar(total=13500*3)
    features = ['haar','lbp','mblbp']
    thresholds = [0,1,2,3,4,5,6,7,8,9]
    mtps=[]
    mtns=[]
    mfps=[]
    mfns=[]
    for each_type in features:
        tps, tns, fps, fns = multiple_threshold_test(each_type,thresholds,mblbpcascade)
        mtps.append(tps)
        mtns.append(tns)
        mfps.append(fps)
        mfns.append(fns)
    return mtps, mtns, mfps, mfns

def run_mblbp(mblbpcascade='./classifier/mblbp-2.xml'):
    progressbar = ProgressBar(total=13500)
    thresholds = [0,1,2,3,4,5,6,7,8,9]
    multiple_threshold_test('mblbp',thresholds,mblbpcascade)

def run_haar():
    progressbar = ProgressBar(total=13500)
    thresholds = [10,11,12,13,14,15,16,17,18,19]
    multiple_threshold_test('haar',thresholds,mblbpcascade)

def draw_pic(mtps,mtns,mfps,mfns,save_name='all.png'):
    # Check the lens of the four values
    line_lens = len(mtps)
    haar_fprs = np.true_divide(mfps[0],(np.add(mfps[0],mtns[0])))
    haar_tprs = np.true_divide(mtps[0],(np.add(mtps[0],mfns[0])))
    lbp_fprs = np.true_divide(mfps[1],(np.add(mfps[1],mtns[1])))
    lbp_tprs = np.true_divide(mtps[1],(np.add(mtps[1],mfns[1])))
    mblbp_fprs = np.true_divide(mfps[2],(np.add(mfps[2],mtns[2])))
    mblbp_tprs = np.true_divide(mtps[2],(np.add(mtps[2],mfns[2])))
    plt.figure()
    plt.plot(haar_fprs,haar_tprs,"g-",label="Haar-Like")
    plt.plot(lbp_fprs,lbp_tprs,"r-",label="LBP")
    plt.plot(mblbp_fprs,mblbp_tprs,"b-",label="MBLBP")
    plt.xlabel("fprs")
    plt.ylabel("tprs")
    plt.title("Haar LBP vs MBLBP")
    plt.legend()
    plt.grid(True)
    plt.savefig('all.png')

if __name__ == '__main__':
    cascades = ['./classifier/mblbp-4.xml']
    for each_cascade in cascades:
        mtps, mtns, mfps, mfns = all_test(mblbpcascade=each_cascade)
        draw_pic(mtps, mtns, mfps, mfns,save_name=each_cascade+'.png')
