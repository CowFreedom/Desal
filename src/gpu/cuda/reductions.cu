#include<stdio.h>
#include "reductions.h"

namespace desal{
	namespace cuda{

	
		__global__
		void k_restrict2h(int m, int k, float2* dest, int pitch_dest, float2* src, int pitch_src){
			int idx=blockIdx.x*blockDim.x+threadIdx.x;
			int idy=blockIdx.y*blockDim.y+threadIdx.y;
			
			dest=(float2*) ((char*)dest+idy*pitch_dest);
			src=(float2*) ((char*)src+2*idy*pitch_src);
			
			for(int i=idy;i<m;i+=gridDim.y*blockDim.y){
				for(int j = idx; j<k; j+=gridDim.x*blockDim.x){
					dest[j].x=src[2*j].x;
					dest[j].y=src[2*j].y;
				}
				dest=(float2*) ((char*)dest+pitch_dest);
				src=(float2*) ((char*)src+2*pitch_src);	
			}
		}

		__global__
		void k_restrict(float hy, float hx, int m, int k, float2* dest, int pitch_dest,cudaTextureObject_t src){
			int idx=blockIdx.x*blockDim.x+threadIdx.x;
			int idy=blockIdx.y*blockDim.y+threadIdx.y;
			
			dest=(float2*) ((char*)dest+idy*pitch_dest);
			
			for(int i=idy;i<m;i+=gridDim.y*blockDim.y){
				for(int j = idx; j<k; j+=gridDim.x*blockDim.x){
					float2 v=tex2D<float2>(src,hx*j+0.5,hy*i+0.5);
					dest[j].x+=v.x;
					dest[j].y+=v.y;													
				}
				dest=(float2*) ((char*)dest+pitch_dest);
			}
		}



	}
}