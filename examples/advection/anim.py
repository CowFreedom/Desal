import math

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
from matplotlib.pyplot import figure
import os


n=810
reps=1

'''
fig, (vax,hax)=plt.subplots(2,1,figsize=(12,6))
path=os.path.dirname(os.path.abspath(__file__))+"/output/output"+str(0)+".csv"
data=np.loadtxt(path,delimiter=",")
vax.imshow(data,cmap=cm.magma)
path=os.path.dirname(os.path.abspath(__file__))+"/output/output"+str(3600)+".csv"
data=np.loadtxt(path,delimiter=",")
hax.imshow(data,cmap=cm.magma)
vax.axis("off")
hax.axis("off")
plt.tight_layout()
'''

data1=np.loadtxt(os.path.dirname(os.path.abspath(__file__))+"/output/output"+str(800)+".csv",delimiter=",")
data2=np.loadtxt(os.path.dirname(os.path.abspath(__file__))+"/output/output"+str(0)+".csv",delimiter=",")
print(np.min(data2))
print(np.max(data2))

frames=[]
for a in range(reps):
	for i in range(n):
		#fig=plt.figure()
		#ax.cla()
		#plt.figure(frameon=False)
		plt.figure(figsize=(0.7, 0.1), dpi=300)
		path=os.path.dirname(os.path.abspath(__file__))+"/output/output"+str(i)+".csv"
		
		data=np.loadtxt(path,delimiter=",")
		frames.append([plt.imshow(data,cmap=cm.magma,animated=True,vmin=np.min(data2), vmax=np.max(data2))])
		
		#c = plt.colorbar()
		#plt.tight_layout()
		plt.axis("off")
		#plt.margins(0)
		#plt.show()
		if i%10 ==0:
			plt.savefig(os.path.dirname(os.path.abspath(__file__))+"/output/animation_almost_shock"+str(i)+".svg",transparent=True)
		plt.close("all")
		
#anim = FuncAnimation(fig, ud, frames=100, interval=100, blit=True)
ani=animation.ArtistAnimation(fig,frames,interval=2,blit=True,repeat_delay=10)
path=os.path.dirname(os.path.abspath(__file__))+"/"

plt.show()
