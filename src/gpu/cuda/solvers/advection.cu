#include <stdio.h>
#include "../error_handling.h"

//static cudaArray* tex_array;
//m_q: Number of vertical interior grid points, k_q: Number of horizontal grid points
namespace desal{
	namespace cuda{
	
		__global__
		void k_advection(float dt, int boundary_padding_thickness, float inv_dy, float inv_dx, int m_q, int k_q, float2* U, int pitch_u, cudaTextureObject_t Q, float* C, int pitch_c){
			m_q-=2*boundary_padding_thickness;
			k_q-=2*boundary_padding_thickness;
			
			int idy=blockIdx.y*blockDim.y+threadIdx.y;
			int idx=blockIdx.x*blockDim.x+threadIdx.x;
		
			float2 p;
			
			float2* U_ptr=(float2*) ((char*)U+(boundary_padding_thickness+idy)*pitch_u);
			float* C_ptr=(float*) ((char*)C+(boundary_padding_thickness+idy)*pitch_c);

			for (int i=idy; i<m_q;i+=gridDim.y*blockDim.y){
				for (int j=idx;j<k_q;j+=gridDim.x*blockDim.x){
					float2 v=U_ptr[j];
					p.x=(j+boundary_padding_thickness+0.5f)-(dt*v.x*inv_dx);
					p.y=(i+boundary_padding_thickness+0.5f)-(dt*v.y*inv_dy);
					float q=tex2D<float>(Q,p.x,p.y);
					C_ptr[j]=q;					
				}		
				C_ptr=(float*) ((char*)C_ptr+gridDim.y*blockDim.y*pitch_c);
				U_ptr=(float2*) ((char*)U_ptr+gridDim.y*blockDim.y*pitch_u);					
			}			
		}
		
		__global__
		void k_advection_field(float dt, int boundary_padding_thickness, float inv_dy, float inv_dx, int m_q, int k_q, float2* U, int pitch_u, cudaTextureObject_t Q, float2* C, int pitch_c){
			m_q-=2*boundary_padding_thickness;
			k_q-=2*boundary_padding_thickness;
			
			int idy=blockIdx.y*blockDim.y+threadIdx.y;
			int idx=blockIdx.x*blockDim.x+threadIdx.x;
		
			float2 p;
			
			float2* U_ptr=(float2*) ((char*)U+(boundary_padding_thickness+idy)*pitch_u)+boundary_padding_thickness;
			float2* C_ptr=(float2*) ((char*)C+(boundary_padding_thickness+idy)*pitch_c)+boundary_padding_thickness;

			for (int i=idy; i<m_q;i+=gridDim.y*blockDim.y){
				for (int j=idx;j<k_q;j+=gridDim.x*blockDim.x){

					float2 v=U_ptr[j];
					p.x=(j+boundary_padding_thickness+0.5f)-(dt*v.x*inv_dx);
					p.y=(i+boundary_padding_thickness+0.5f)-(dt*v.y*inv_dy);
					float2 q=tex2D<float2>(Q,p.x,p.y);
					C_ptr[j].x=q.x;
					C_ptr[j].y=q.y;					
				}	
				C_ptr=(float2*) ((char*)C_ptr+gridDim.y*blockDim.y*pitch_c);
				U_ptr=(float2*) ((char*)U_ptr+gridDim.y*blockDim.y*pitch_u);		
			}
		}
	

	}
}
