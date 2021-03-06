"""Plot initial condition and last aproximation.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
DIR_BASE = sys.argv[1]
sim = sys.argv[2]
U00 = np.loadtxt(DIR_BASE + "U0_" + sim + ".txt")
B00 = np.loadtxt(DIR_BASE + "B0_" + sim + ".txt")
U0 = U00.reshape((128, 128))
B0 = B00.reshape((128, 128))
Uaa = np.loadtxt(DIR_BASE + "U_" + sim + ".txt")
Baa = np.loadtxt(DIR_BASE + "B_" + sim + ".txt")
Ua = Uaa.reshape((128, 128))
Ba = Baa.reshape((128, 128))
plt.figure(figsize=(8, 6))
plt.subplot(2, 2, 1)
plt.imshow(U0, origin="lower", 
    extent=[0, 90, 0, 90], cmap=plt.cm.jet)
plt.colorbar()
#plt.show()
plt.subplot(2, 2, 2)
plt.imshow(B0, origin="lower", 
    extent=[0, 90, 0, 90], cmap=plt.cm.Oranges)
plt.colorbar()
#plt.show()
plt.subplot(2, 2, 3)
plt.imshow(Ua, origin="lower", 
    extent=[0, 90, 0, 90], cmap=plt.cm.jet)
plt.colorbar()
#plt.show()
plt.subplot(2, 2, 4)
plt.imshow(Ba, origin="lower",  
    extent=[0, 90, 0, 90], cmap=plt.cm.Oranges)
plt.colorbar()
plt.tight_layout()
plt.show()
#print(np.all(U0 == Ua) and np.all(B0 == Ba))
print(np.all(Ua[0] == 0), np.all(Ua[-1] == 0), np.all(Ua[:,0] == 0), np.all(Ua[:,-1] == 0))