#pragma once

namespace desal{
	
	namespace cuda{

		namespace transformations{
			namespace blocksizes{
				namespace group1{
					constexpr int MX=256;
					constexpr int MY=4;
				}		
			}	
		}
		
		template<class F, class F2>
		__host__
		cudaError_t transform_entries_into_residuals_device(F alpha, F beta, int boundary_padding_thickness, int m, int k, F2* A_d, int pitch_a, F2* B_d, int pitch_b, F2* r_d, int pitch_r);


		__host__
		cudaError_t transform_entries_into_residuals_device(float alpha,float beta, int boundary_padding_thickness, int m, int k, float2* A_d, int pitch_a, float2* B_d, int pitch_b, float2* r_d, int pitch_r);

		__global__
		void k_prolong_by_interpolation_and_add_2h(int m, int k, float2* dest, int pitch_dest, cudaTextureObject_t src);

		__global__
		void k_prolong_by_interpolation_and_add(float hy, float hx, int m, int k, float2* dest, int pitch_dest, cudaTextureObject_t src);

		template<class F2>
		__host__
		cudaError_t prolong_by_interpolation_and_add_2h(int m_p, int k_p, int m_r, int k_r, F2* dest, int pitch_dest, F2* src, int pitch_src){
			//Create Resource descriptions
			cudaResourceDesc resDesc;
			memset(&resDesc,0,sizeof(resDesc));

			resDesc.resType = cudaResourceTypePitch2D;
			resDesc.res.pitch2D.devPtr=src;
			resDesc.res.pitch2D.width=m_r;
			resDesc.res.pitch2D.height=m_r;
			resDesc.res.pitch2D.pitchInBytes=pitch_src;
			resDesc.res.pitch2D.desc=cudaCreateChannelDesc<F2>(); 

			//Create Texture description
			cudaTextureDesc texDesc;
			memset(&texDesc,0,sizeof(texDesc));
			texDesc.normalizedCoords = false;
			texDesc.filterMode = cudaFilterModeLinear;
			texDesc.readMode=cudaReadModeElementType;
			texDesc.addressMode[0] = cudaAddressModeBorder;
			texDesc.addressMode[1] = cudaAddressModeBorder;

			cudaTextureObject_t src_tex;
			
			cudaError_t err=cudaCreateTextureObject(&src_tex, &resDesc, &texDesc, NULL);
			
			if (err !=cudaSuccess){
				return err;
			}
			
			int threads_per_block_x=transformations::blocksizes::group1::MX;	
			int threads_per_block_y=transformations::blocksizes::group1::MY;	
			int blocks_x=ceil(static_cast<float>(k_p)/(threads_per_block_x));
			int blocks_y=ceil(static_cast<float>(m_p)/(threads_per_block_y));
			
			dim3 threads=dim3(threads_per_block_x,threads_per_block_y,1);
			dim3 blocks=dim3(blocks_x,blocks_y,1);
			k_prolong_by_interpolation_and_add_2h<<<blocks,threads>>>(m_p,k_p, dest, pitch_dest, src_tex);
			return cudaSuccess;
		}

		//Prolongs source 2D array and adds it to destination
		template<class F, class F2>
		__host__
		cudaError_t prolong_by_interpolation_and_add(F hy, F hx, int m_p, int k_p, int m_r, int k_r, F2* dest, int pitch_dest, F2* src, int pitch_src){
			//Create Resource descriptions
			cudaResourceDesc resDesc;
			memset(&resDesc,0,sizeof(resDesc));

			resDesc.resType = cudaResourceTypePitch2D;
			resDesc.res.pitch2D.devPtr=src;
			resDesc.res.pitch2D.width=k_r;
			resDesc.res.pitch2D.height=m_r;
			resDesc.res.pitch2D.pitchInBytes=pitch_src;
			resDesc.res.pitch2D.desc=cudaCreateChannelDesc<F2>(); 

			//Create Texture description
			cudaTextureDesc texDesc;
			memset(&texDesc,0,sizeof(texDesc));
			texDesc.normalizedCoords = false;
			texDesc.filterMode = cudaFilterModeLinear;
			texDesc.readMode=cudaReadModeElementType;
			texDesc.addressMode[0] = cudaAddressModeBorder;
			texDesc.addressMode[1] = cudaAddressModeBorder;

			cudaTextureObject_t src_tex;
			cudaError_t err=cudaCreateTextureObject(&src_tex, &resDesc, &texDesc, NULL);
			
			if (err !=cudaSuccess){
				return err;
			}	
			
			int threads_per_block_x=transformations::blocksizes::group1::MX;	
			int threads_per_block_y=transformations::blocksizes::group1::MY;	
			int blocks_x=ceil(static_cast<float>(k_p)/(threads_per_block_x));
			int blocks_y=ceil(static_cast<float>(m_p)/(threads_per_block_y));
			
			dim3 threads=dim3(threads_per_block_x,threads_per_block_y,1);
			dim3 blocks=dim3(blocks_x,blocks_y,1);

			k_prolong_by_interpolation_and_add<<<blocks,threads>>>(hy,hx,m_p,k_p, dest, pitch_dest, src_tex);
			return cudaSuccess;
		}

		template<class F, class F2>
		__host__
		cudaError_t prolong_and_add(int m_p, int k_p, int m_r, int k_r, float2* dest, int pitch_dest, float2* src, int pitch_src){
			if ((((m_p)%2)!=0)&&((k_p%2)!=0)){
				return prolong_by_interpolation_and_add_2h<F2>(m_p, k_p, m_r, k_r, dest, pitch_dest,src,pitch_src);	
			}
			else{
				float hy=static_cast<float>(m_r)/(m_p-1);
				float hx=static_cast<float>(k_r)/(k_p-1);
				return prolong_by_interpolation_and_add<F,F2>(hy,hx,m_p,k_p,m_r,k_r, dest, pitch_dest,src,pitch_src);	
			}	
		}

		template<class F, class F2>
		cudaError_t divergence(F dy, F dx, int boundary_padding_thickness, int m, int k,F2* A_d, size_t pitch_a, F2* C_d, size_t pitch_c);
		
		template<class F, class F2>
		cudaError_t subtract_gradient(F dy, F dx, int boundary_padding_thickness, int m, int k,F2* A_d, size_t pitch_a, F2* C_d, size_t pitch_c);
	}
}