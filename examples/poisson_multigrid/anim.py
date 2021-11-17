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



fig=plt.figure()

frames=[]
for a in range(reps):

	for i in range(0,n):
		#ax.cla()

		path=os.path.dirname(os.path.abspath(__file__))+"/output/output"+str(i)+".csv"
		data=np.loadtxt(path,delimiter=",")
		im1=plt.imshow(data,cmap=cm.magma,animated=True,vmin=0, vmax=7)
		frames.append([im1])


		
#anim = FuncAnimation(fig, ud, frames=100, interval=100, blit=True)
ani=animation.ArtistAnimation(fig,frames,interval=2,blit=True,repeat_delay=10)
path=os.path.dirname(os.path.abspath(__file__))+"/"
ani.save(path+"movie.gif")

'''
fig=plt.figure()

for i in range(n):
	if i% 100 ==0:
		fig =plt.figure(figsize=(16,12))
		vax=fig.add_subplot(211)
		path=os.path.dirname(os.path.abspath(__file__))+"/output/output"+str(i)+".csv"	
		data=np.loadtxt(path,delimiter=",")
		print(np.max(data))
		im1=vax.imshow(data,cmap=cm.magma,vmin=0, vmax=10)
		divider=make_axes_locatable(vax)
		cax=divider.append_axes('right', size='5%',pad=0.05)
		fig.colorbar(im1,cax=cax, orientation="vertical")
		hax=fig.add_subplot(212)

		im2=hax.plot([0.0125*x for x in range(0,n)],[np.sin(0.5*0.0125*x) for x in range(0,n)])
		im2=hax.plot([0.0125*i],[np.sin(0.5*0.0125*i)],marker="o")
		divider=make_axes_locatable(hax)
		#cax=divider.append_axes('up', size='5%',pad=0.05)
		#fig.colorbar(im2,cax=cax, orientation="vertical")
		
		#vax.axis("off")
		#hax.axis("off")
		plt.tight_layout()

		#plt.savefig(os.path.dirname(os.path.abspath(__file__))+"/output/animation_poisson_changing_boundary"+str(i)+".svg",transparent=True)
		plt.close()

'''
plt.show()
