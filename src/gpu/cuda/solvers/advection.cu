#include <stdio.h>
#include "../error_handling.h"

//static cudaArray* tex_array;
//m_q: Number of vertical interior grid points, k_q: Number of horizontal grid points
namespace desal{
	namespace cuda{
		
		__global__
		void k_advection(float dt, int boundary_padding_thickness, float inv_dy, float inv_dx, int m_q, int k_q, float2* U, int pitch_u, cudaTextureObject_t Q, float2* C, int pitch_c){
			m_q-=boundary_padding_thickness;
			k_q-=boundary_padding_thickness;
			
			int idy=blockIdx.y*blockDim.y+threadIdx.y+boundary_padding_thickness;
			int idx=blockIdx.x*blockDim.x+threadIdx.x+boundary_padding_thickness;
		
			float2 p;
			
			U=(float2*) ((char*)U+idy*pitch_u);
			C=(float2*) ((char*)C+idy*pitch_c);

			for (int i=idy; i<m_q;i+=gridDim.y*blockDim.y){
				for (int j=idx;j<k_q;j+=gridDim.x*blockDim.x){
					float2 v=U[j];
					p.x=(j+0.5f)-(dt*v.x*inv_dx);
					p.y=(i+0.5f)-(dt*v.y*inv_dy);
					float2 q=tex2D<float2>(Q,p.x,p.y);
					C[j].x=q.x;
					C[j].y=q.y;					
				}	
				C=(float2*) ((char*)C+gridDim.y*blockDim.y*pitch_c);
				U=(float2*) ((char*)U+gridDim.y*blockDim.y*pitch_u);		
			}
		}

	}
}
