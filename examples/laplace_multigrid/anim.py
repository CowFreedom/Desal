import math

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os


n=3599
reps=1

path=os.path.dirname(os.path.abspath(__file__))+"/output/final_heat_distribution.csv"
data=np.loadtxt(path,delimiter=",")

fig =plt.figure(figsize=(16,12))
vax=fig.add_subplot(121)

im1=vax.imshow(data,cmap=cm.magma,vmin=0, vmax=25)
divider=make_axes_locatable(vax)
cax=divider.append_axes('right', size='5%',pad=0.05)
fig.colorbar(im1,cax=cax, orientation="vertical")
hax=fig.add_subplot(122)
print(np.max(data))
im2=hax.imshow(data,cmap=cm.magma,vmin=np.min(data), vmax=np.max(data))
divider=make_axes_locatable(hax)
cax=divider.append_axes('right', size='5%',pad=0.05)
fig.colorbar(im2,cax=cax, orientation="vertical")

vax.axis("off")
hax.axis("off")
plt.tight_layout()
plt.savefig(os.path.dirname(os.path.abspath(__file__))+"/output/final_heat_distribution.svg",transparent=True)
plt.show()
