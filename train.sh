opencv_traincascade -data haar -vec trainingfaces_24-24.vec -bg negatives.txt\
   -numStages 12 -minHitRate 0.999 -maxFalseAlarmRate 0.5 -numPos 16000 -featureType HAAR\
   -numNeg 10000 -w 24 -h 24 -precalcValBufSize 1024\
   -precalcIdxBufSize 1024
