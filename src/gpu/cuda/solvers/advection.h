#pragma once

namespace desal{
	namespace cuda{
		
		__global__
		void k_advection(float dt, int boundary_padding_thickness, float inv_dy, float inv_dx, int m_q, int k_q, float2* U, int pitch_u, cudaTextureObject_t Q, float* C, int pitch_c);
	
		
		__global__
		void k_advection_field(float dt, int boundary_padding_thickness, float inv_dy, float inv_dx, int m_q, int k_q, float2* U, int pitch_u, cudaTextureObject_t Q, float2* C, int pitch_c);
		
		template<class F, class F2>
		__host__
		DesalStatus advection_field(F dt, int boundary_padding_thickness, F dy, F dx,  int m_q,  int k_q, F2* U_d, int pitch_u, F2* Q_d, int pitch_q, F2* C_d, int pitch_c){
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
				return DesalStatus::CUDAError;
			}
			
			int threads_per_block_x=8;	
			int threads_per_block_y=4;	
			int blocks_x=ceil(static_cast<F>(m_q-2)/(threads_per_block_x));
			int blocks_y=ceil(static_cast<F>(k_q-2)/(threads_per_block_y));
			
			dim3 threads=dim3(threads_per_block_x,threads_per_block_y,1);
			dim3 blocks=dim3(blocks_x,blocks_y,1);

			k_advection_field_2D<F,F2><<<blocks,threads>>>(dt,boundary_padding_thickness,1/dy,1/dx,m_q,k_q,U_d,pitch_u,Q_tex,C_d,pitch_c);
			return DesalStatus::Success;
		}
		
		template
		DesalStatus advection_field_2D_device(float dt, int boundary_padding_thickness, float dy, float dx,  int m_q,  int k_q, float2* U_d, int pitch_u, float2* Q_d, int pitch_q, float2* C_d, int pitch_c);

		template<class F, class F2>
		__host__
		DesalStatus advection(F dt, int boundary_padding_thickness, F dy, F dx,  int m_q,  int k_q, F2* U_d, int pitch_u, F* Q_d, int pitch_q, F* C_d, int pitch_c){
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
			
			int threads_per_block_x=8;	
			int threads_per_block_y=4;	
			int blocks_x=ceil(static_cast<F>(m_q-2)/(threads_per_block_x));
			int blocks_y=ceil(static_cast<F>(k_q-2)/(threads_per_block_y));
			
			dim3 threads=dim3(threads_per_block_x,threads_per_block_y,1);
			dim3 blocks=dim3(blocks_x,blocks_y,1);

			k_advection<<<blocks,threads>>>(dt,boundary_padding_thickness,1/dy,1/dx,m_q,k_q,U_d,pitch_u,Q_tex,C_d,pitch_c);
			return DesalStatus::Success;
		}
	}
}