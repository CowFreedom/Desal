import math

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os


n=400
reps=1


fig=plt.figure()

frames=[]
for a in range(reps):
	i=0
	while True:
		#ax.cla()

		path=os.path.dirname(os.path.abspath(__file__))+"/output/output"+str(i)+".csv"
		
		try:
			data=np.loadtxt(path,delimiter=",")
		except:
			break
		data=np.loadtxt(path,delimiter=",")
		im1=plt.imshow(data,cmap=cm.magma,animated=True,vmin=0, vmax=25)
		frames.append([im1])
		i+=1


		
#anim = FuncAnimation(fig, ud, frames=100, interval=100, blit=True)
ani=animation.ArtistAnimation(fig,frames,interval=2,blit=True,repeat_delay=10)
path=os.path.dirname(os.path.abspath(__file__))+"/"
ani.save(path+"movie.gif")
'''

fig=plt.figure()

frames=[]
for a in range(reps):
	i=0
	while True:
		#ax.cla()

		path=os.path.dirname(os.path.abspath(__file__))+"/output/output"+str(i)+".csv"
		try:
			data=np.loadtxt(path,delimiter=",")
		except:
			break
		data=np.loadtxt(path,delimiter=",")
		fig=plt.figure(figsize=(1,1))
		plt.axis("off")
		im1=plt.imshow(data,cmap=cm.magma,animated=True,vmin=0, vmax=25)
		print("Saving image "+str(i))
		plt.savefig(os.path.dirname(os.path.abspath(__file__))+"/img/test"+str(i)+".png")
		i+=1
		

	


'''
plt.show()
