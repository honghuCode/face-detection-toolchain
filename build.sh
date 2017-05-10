find ./positive_images -iname "*.jpg" > positives.txt
find ./negative_images -iname "*.jpg" > negatives.txt
perl bin/createsamples.pl positives.txt negatives.txt samples 1500\
   "opencv_createsamples -bgcolor 0 -bgthresh 0 -maxxangle 1.1\
   -maxyangle 1.1 maxzangle 0.5 -maxidev 40 -w 80 -h 40"
python ./tools/mergevec.py -v samples/ -o samples.vec
