#include <stdio.h>
#include "../error_handling.h"

//static cudaArray* tex_array;
//m_q: Number of vertical interior grid points, k_q: Number of horizontal grid points
namespace desal{
	namespace cuda{
	
		template<class F, class F2>
		__global__
		void k_advection_2D(F dt, int boundary_padding_thickness, F dy, F dx, int m_q, int k_q, F2* U, int pitch_u, cudaTextureObject_t Q, F* C, int pitch_c){
			m_q-=2*boundary_padding_thickness;
			k_q-=2*boundary_padding_thickness;
			
			int idy=blockIdx.y*blockDim.y+threadIdx.y;
			int idx=blockIdx.x*blockDim.x+threadIdx.x;
		
			F2 p;
			
			F2* U_ptr=(F2*) ((char*)U+(boundary_padding_thickness+idy)*pitch_u);
			F* C_ptr=(F*) ((char*)C+(boundary_padding_thickness+idy)*pitch_c);

			for (int i=idy; i<m_q;i+=gridDim.y*blockDim.y){
				for (int j=idx;j<k_q;j+=gridDim.x*blockDim.x){

					F2 v=U_ptr[j];
					p.x=(j+boundary_padding_thickness+0.5f)-(dt*v.x*dx);
					p.y=(i+boundary_padding_thickness+0.5f)-(dt*v.y*dy);
					F q=tex2D<F>(Q,p.x,p.y);
					C_ptr[j]=q;					
				}		
				C_ptr=(F*) ((char*)C_ptr+gridDim.y*blockDim.y*pitch_c);
				U_ptr=(F2*) ((char*)U_ptr+gridDim.y*blockDim.y*pitch_u);					
			}	
		
		}
		
		template<class F, class F2>
		__global__
		void k_advection_field_2D(F dt, int boundary_padding_thickness, F dy, float dx, int m_q, int k_q, F2* U, int pitch_u, cudaTextureObject_t Q, F2* C, int pitch_c){
			m_q-=2*boundary_padding_thickness;
			k_q-=2*boundary_padding_thickness;
			
			int idy=blockIdx.y*blockDim.y+threadIdx.y;
			int idx=blockIdx.x*blockDim.x+threadIdx.x;
		
			F2 p;
			
			F2* U_ptr=(F2*) ((char*)U+(boundary_padding_thickness+idy)*pitch_u)+boundary_padding_thickness;
			F2* C_ptr=(F2*) ((char*)C+(boundary_padding_thickness+idy)*pitch_c)+boundary_padding_thickness;

			for (int i=idy; i<m_q;i+=gridDim.y*blockDim.y){
				for (int j=idx;j<(k_q-1);j+=gridDim.x*blockDim.x){

					F2 v=U_ptr[j];
					p.x=(j+boundary_padding_thickness+0.5f)-(dt*v.x*dx);
					p.y=(i+boundary_padding_thickness+0.5f)-(dt*v.y*dy);
					F2 q=tex2D<F2>(Q,p.x,p.y);
					C_ptr[j].x=q.x;
					C_ptr[j].y=q.y;					
				}	
				C_ptr=(F2*) ((char*)C_ptr+gridDim.y*blockDim.y*pitch_c);
				U_ptr=(F2*) ((char*)U_ptr+gridDim.y*blockDim.y*pitch_u);		
			}
		}
	
		template<class F, class F2>
		__host__
		DesalStatus advection_field_2D_device(F dt, int boundary_padding_thickness, F dy, F dx,  int m_q,  int k_q, F2* U_d, int pitch_u, F2* Q_d, int pitch_q, F2* C_d, int pitch_c){
			if ((m_q<3) || (k_q<3)){
				return DesalStatus::InvalidParameters;
			}

			//Create Resource description
			cudaResourceDesc resDesc;
			memset(&resDesc,0,sizeof(resDesc));

			resDesc.resType = cudaResourceTypePitch2D;
			resDesc.res.pitch2D.devPtr=Q_d;
			resDesc.res.pitch2D.width=k_q;
			resDesc.res.pitch2D.height=m_q;
			resDesc.res.pitch2D.pitchInBytes=pitch_q;
			resDesc.res.pitch2D.desc=cudaCreateChannelDesc<F2>(); //is equivalent to cudaCreateChannelDesc<F>()
			
			/*
			resDesc.res.pitch2D.desc=cudaCreateChannelDesc(32,32,0,0,cudaChannelFormatKindFloat); //is equivalent to cudaCreateChannelDesc<float2>()
		*/
			//Create Texture description
			cudaTextureDesc texDesc;
			memset(&texDesc,0,sizeof(texDesc));
			texDesc.normalizedCoords = false;
			texDesc.filterMode = cudaFilterModeLinear;
			texDesc.readMode=cudaReadModeElementType;
			texDesc.addressMode[0] = cudaAddressModeClamp;
			texDesc.addressMode[1] = cudaAddressModeClamp;

			//Create Texture Object
			cudaTextureObject_t Q_tex;
			cudaError_t error1=cudaCreateTextureObject(&Q_tex, &resDesc, &texDesc, NULL);
			if (error1 !=cudaSuccess){
			printf("error hier\n");
				return DesalStatus::CUDAError;
			}
			
			int threads_per_block_x=8;	
			int threads_per_block_y=4;	
			int blocks_x=ceil(static_cast<F>(m_q-2)/(threads_per_block_x));
			int blocks_y=ceil(static_cast<F>(k_q-2)/(threads_per_block_y));
			
			dim3 threads=dim3(threads_per_block_x,threads_per_block_y,1);
			dim3 blocks=dim3(blocks_x,blocks_y,1);

			k_advection_field_2D<F,F2><<<blocks,threads>>>(dt,boundary_padding_thickness,dy,dx,m_q,k_q,U_d,pitch_u,Q_tex,C_d,pitch_c);
			return DesalStatus::Success;
		}
		
		template
		DesalStatus advection_field_2D_device(float dt, int boundary_padding_thickness, float dy, float dx,  int m_q,  int k_q, float2* U_d, int pitch_u, float2* Q_d, int pitch_q, float2* C_d, int pitch_c);

		template<class F, class F2>
		__host__
		DesalStatus advection_2D_f32_device(F dt, int boundary_padding_thickness, F dy, F dx,  int m_q,  int k_q, F2* U_d, int pitch_u, F* Q_d, int pitch_q, F* C_d, int pitch_c){
			if ((m_q<3) || (k_q<3)){
				return DesalStatus::InvalidParameters;
			}

			//Create Resource description
			cudaResourceDesc resDesc;
			memset(&resDesc,0,sizeof(resDesc));

			resDesc.resType = cudaResourceTypePitch2D;
			resDesc.res.pitch2D.devPtr=Q_d;
			resDesc.res.pitch2D.width=k_q;
			resDesc.res.pitch2D.height=m_q;
			resDesc.res.pitch2D.pitchInBytes=pitch_q;
			resDesc.res.pitch2D.desc=cudaCreateChannelDesc<F>(); //is equivalent to cudaCreateChannelDesc<F>()
			
			/*
			resDesc.res.pitch2D.desc=cudaCreateChannelDesc(32,32,0,0,cudaChannelFormatKindFloat); //is equivalent to cudaCreateChannelDesc<float2>()
		*/
			//Create Texture description
			cudaTextureDesc texDesc;
			memset(&texDesc,0,sizeof(texDesc));
			texDesc.normalizedCoords = false;
			texDesc.filterMode = cudaFilterModeLinear;
			texDesc.readMode=cudaReadModeElementType;
			texDesc.addressMode[0] = cudaAddressModeClamp;
			texDesc.addressMode[1] = cudaAddressModeClamp;

			//Create Texture Object
			cudaTextureObject_t Q_tex;
			cudaError_t error1=cudaCreateTextureObject(&Q_tex, &resDesc, &texDesc, NULL);
			if (error1 !=cudaSuccess){
				return DesalStatus::CUDAError;
			}
			printf("w, h: %d,%d\n",k_q,m_q);
			
			int threads_per_block_x=8;	
			int threads_per_block_y=4;	
			int blocks_x=ceil(static_cast<F>(m_q-2)/(threads_per_block_x));
			int blocks_y=ceil(static_cast<F>(k_q-2)/(threads_per_block_y));
			
			dim3 threads=dim3(threads_per_block_x,threads_per_block_y,1);
			dim3 blocks=dim3(blocks_x,blocks_y,1);

			k_advection_2D<F,F2><<<blocks,threads>>>(dt,boundary_padding_thickness,dy,dx,m_q,k_q,U_d,pitch_u,Q_tex,C_d,pitch_c);
			return DesalStatus::Success;
		}
	}
}
