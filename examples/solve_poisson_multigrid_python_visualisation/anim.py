import math

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
import os


n=3599
reps=1


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
fig=plt.figure()
frames=[]
for a in range(reps):
	for i in range(n):
		#ax.cla()

		path=os.path.dirname(os.path.abspath(__file__))+"/output/output"+str(i)+".csv"
		data=np.loadtxt(path,delimiter=",")
		frames.append([plt.imshow(data,cmap=cm.magma,animated=True)])
		
#anim = FuncAnimation(fig, ud, frames=100, interval=100, blit=True)
ani=animation.ArtistAnimation(fig,frames,interval=2,blit=True,repeat_delay=10)
path=os.path.dirname(os.path.abspath(__file__))+"/"
ani.save(path+"movie.gif")
'''
plt.show()
