import numpy as np
import numpy as np 
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

haar_tp = [450,448,446,445,445,445,444,444,444,441]
haar_tn = [502,512,520,525,529,532,536,537,546,546]
haar_fp = []
haar_fn = []

lbp_tp = [449,449,448,447,446,444,443,443,443,443]
lbp_tn = [227,494,530,545,545,554,554,555,555,555]
lbp_fp = []
lbp_fn = []

mblbp_tp = [442,442,429,421,414,405,393,386,380,374]
mblbp_tn = [511,511,566,574,579,585,587,593,598,601]
mblbp_fp = []
mblbp_fn = []

for index,each in enumerate(haar_tp):
    haar_fn.append(450-each)

for index,each in enumerate(lbp_tp):
    lbp_fn.append(450-each)

for index,each in enumerate(mblbp_tp):
    mblbp_fn.append(450-each)

for index,each in enumerate(haar_tn):
    haar_fp.append(900-each)

for index,each in enumerate(lbp_tn):
    lbp_fp.append(900-each)

for index,each in enumerate(mblbp_tn):
    mblbp_fp.append(900-each)

haar_fprs = np.true_divide(haar_fp,(np.add(haar_fp,haar_tn)))
# haar_fprs=haar_fprs[::-1]
haar_tprs = np.true_divide(haar_tp,(np.add(haar_tp,haar_fn)))
lbp_fprs = np.true_divide(lbp_fp,(np.add(lbp_fp,lbp_tn)))
lbp_tprs = np.true_divide(lbp_tp,(np.add(lbp_tp,lbp_fn)))
mblbp_fprs = np.true_divide(mblbp_fp,(np.add(mblbp_fp,mblbp_tn)))
mblbp_tprs = np.true_divide(mblbp_tp,(np.add(mblbp_tp,mblbp_fn)))



plt.figure()
plt.plot(haar_fprs,haar_tprs,"g-",label="Haar-Like")
plt.plot(lbp_fprs,lbp_tprs,"r-",label="LBP")
plt.plot(mblbp_fprs,mblbp_tprs,"b-",label="MBLBP")
plt.xlabel("fprs")
plt.ylabel("tprs")
plt.title("Haar vs LBP")
plt.legend()
plt.grid(True)
plt.savefig('all.png')
