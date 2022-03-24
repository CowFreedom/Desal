import numpy as np
from matplotlib import pyplot
import matplotlib.pyplot as plt

def J_v_al_ds(p_draw,p_feed,Km,A,B):
	f=lambda x: Km*np.log((A*p_draw-x+B)/(A*p_feed+B)) -x
	f_p=lambda x:Km*(1/((A*p_draw-x+B)/(A*p_feed+B)))*(-1/(A*p_feed+B))-1
#	f=lambda x: ((A*p_draw-x+B)/(A*p_feed+B))**Km-np.exp(x)
#	f_p=lambda x:-np.exp(x)+((A*p_draw-x+B)/(A*p_feed+B))**(Km-1)*-(1.0/(A*p_feed+B))

	start=(178*(10**-3))/3600
	#start=0
	x_n=start
#	print((A*p_draw-x_n+B))
#	input()

	x_prev=1
	eps=1e-23
	iter=0
	while iter<14:
		x_prev=x_n

		x_n=x_n-f(x_n)/f_p(x_n)	
		#print(f(x_n)/f_p(x_n))
		#input()
		iter+=1

	#print(np.exp(x_n-Km))	
		
	#print(f(x_n))
	return x_n
	
def J_v_al_fs(p_draw,p_feed,Km,A,B):
	f=lambda x: Km*np.log((A*p_draw+B)/(A*p_feed+x+B)) -x
	f_p=lambda x:Km*(1/((A*p_draw+B)/(A*p_feed+x+B)))*-(A*p_draw+B)*(A*p_feed+x+B)**-2-1
	#f=lambda x: (A*p_draw-x+B)-np.exp(x-Km)*(A*p_feed+B) 
	#f_p=lambda x:-1-np.exp(x-Km)*(A*p_feed+B) 

	#start=(1*(10**-3))/3600
	start=0
	x_n=start
#	print((A*p_draw-x_n+B))
#	input()
	x_prev=1
	eps=1e-23
	iter=0
	while iter<6:
		x_prev=x_n
		x_n=x_n-f(x_n)/f_p(x_n)	
		iter+=1

	#print(np.exp(x_n-Km))	
		
	#print(f(x_n))
	return x_n	
	
D=1.3*(10**-9)
p_draw=99.9*10**(5)
p_feed=0.45*10**(5)
A=5*(10**-12)
B=1*(10**-7)

n=1000
S_al_ds=np.zeros(n)
J_vs_al_ds=np.zeros(n)
J_vs_al_fs=np.zeros(n)

for i in range(1,n+1):
	S=(i*1)/n
	#print(S)
	S_al_ds[i-1]=S
	S/=1000 #mm zu m
	Km=D/S	
	#print(Km)
	#print(D/S)
	J_vs_al_ds[i-1]=J_v_al_ds(p_draw,p_feed,Km,A,B)
	J_vs_al_fs[i-1]=J_v_al_fs(p_draw,p_feed,Km,A,B)
	
plt.plot(S_al_ds,J_vs_al_ds*1000*3600,label="AL:DS")
plt.plot(S_al_ds,J_vs_al_fs*1000*3600,label="AL:FS")
plt.xlabel("Membrane Structural parameter S in mm")
plt.ylabel(r'Water flow in $\mathrm{\frac{L}{m^2h}}$')
#plt.set_label()
print(J_vs_al_ds*1000*3600)
plt.show()