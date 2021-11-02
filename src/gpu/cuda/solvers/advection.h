#pragma once

namespace desal{
	namespace cuda{
	
		namespace constants{
			namespace blocksizes_advection{			
				namespace group1{
					constexpr int MX=8;
					constexpr int MY=4;		
				}
			}	
		}
	
		__global__
		void k_advection(float dt, int boundary_padding_thickness, float inv_dy, float inv_dx, int m_q, int k_q, float2* U, int pitch_u, cudaTextureObject_t Q, float* C, int pitch_c);
	
		
		__global__
		void k_advection(float dt, int boundary_padding_thickness, float inv_dy, float inv_dx, int m_q, int k_q, float2* U, int pitch_u, cudaTextureObject_t Q, float2* C, int pitch_c);
		
	
		template<class F, class F2>
		__host__
		cudaError_t advection(F dt, int boundary_padding_thickness, F dy, F dx,  int m_q,  int k_q, F2* U_d, int pitch_u, F2* Q_d, int pitch_q, F2* C_d, int pitch_c){
			if ((m_q<3) || (k_q<3) | (boundary_padding_thickness<0)){
				return cudaErrorInvalidValue;
			}

			//Create Resource description
			cudaResourceDesc resDesc;
			memset(&resDesc,0,sizeof(resDesc));

			resDesc.resType = cudaResourceTypePitch2D;
			resDesc.res.pitch2D.devPtr=Q_d;
			resDesc.res.pitch2D.width=k_q;
			resDesc.res.pitch2D.height=m_q;
			resDesc.res.pitch2D.pitchInBytes=pitch_q;
			resDesc.res.pitch2D.desc=cudaCreateChannelDesc<F2>(); 
			
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
			cudaError_t error=cudaCreateTextureObject(&Q_tex, &resDesc, &texDesc, NULL);
			
			if (error !=cudaSuccess){
				return error;
			}	
			
			int threads_per_block_x=constants::blocksizes_advection::group1::MX;	
			int threads_per_block_y=constants::blocksizes_advection::group1::MY;	
			int blocks_x=ceil(static_cast<F>(m_q-2)/(threads_per_block_x));
			int blocks_y=ceil(static_cast<F>(k_q-2)/(threads_per_block_y));
			
			dim3 threads=dim3(threads_per_block_x,threads_per_block_y,1);
			dim3 blocks=dim3(blocks_x,blocks_y,1);

			k_advection<<<blocks,threads>>>(dt,boundary_padding_thickness,1.0/dy,1.0/dx,m_q,k_q,U_d,pitch_u,Q_tex,C_d,pitch_c);
			return cudaSuccess;
		}
	}
}