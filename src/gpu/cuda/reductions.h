#pragma once

namespace desal{
	namespace cuda{
		
		namespace reductions{
			namespace blocksizes{
				namespace group1{
					constexpr int MX=512;
					constexpr int MY=2;
				}
				namespace group2{
					constexpr int MX=256;
					constexpr int MY=4;
				}
				namespace power_of_two_and_below_1024{
					constexpr int MX=2;
				}					
				namespace power_of_two_and_jointly_below_1024{
					constexpr int MX=8;
					constexpr int MY=4;
				}					
			}	
		}
		

		template<unsigned int THREADS_X_PER_BLOCK, class F>
		__device__
		void k_warp_reduce_sum_2D(volatile F* sdata, int ty, int tx){
			int index=ty*THREADS_X_PER_BLOCK+tx;
			
			if (THREADS_X_PER_BLOCK >=64){
				sdata[index]+=sdata[index+32];
				__syncthreads();
			}

			if (THREADS_X_PER_BLOCK >=32){
				sdata[index]+=sdata[index+16];
				__syncthreads();
			}
			if (THREADS_X_PER_BLOCK >=16){
				sdata[index]+=sdata[index+8];
				__syncthreads();
			}
			if (THREADS_X_PER_BLOCK >=8){
				sdata[index]+=sdata[index+4];
				__syncthreads();
			}
			if (THREADS_X_PER_BLOCK >=4){
				sdata[index]+=sdata[index+2];
				__syncthreads();
			}
			
			if (THREADS_X_PER_BLOCK >=2){
				sdata[index]+=sdata[index+1];
				__syncthreads();
			}
		}

		template<unsigned int THREADS_X_PER_BLOCK, class F>
		__device__
		void k_warp_reduce_sum(volatile F* sdata, int tx){

			if(THREADS_X_PER_BLOCK>=64){
				sdata[tx]+=sdata[tx+32];	
			}
			if(THREADS_X_PER_BLOCK>=32){
				sdata[tx]+=sdata[tx+16];	
			}
			if(THREADS_X_PER_BLOCK>=16){
				sdata[tx]+=sdata[tx+8];	
			}
			if(THREADS_X_PER_BLOCK>=8){
			sdata[tx]+=sdata[tx+4];	
			}
			if(THREADS_X_PER_BLOCK>=4){
				sdata[tx]+=sdata[tx+2];	
			}
			if(THREADS_X_PER_BLOCK>=2){
				sdata[tx]+=sdata[tx+1];	
			}
		}

		/*Reduces an array with n elements to log_b(n) its size by summing its entries, whereas b is the number of blocks in the grid.*/
		template<unsigned int THREADS_X_PER_BLOCK, class F>
		__global__
		void k_reduce_sum(int n, F* r, int stride_r){
			if (n<blockIdx.x*2*blockDim.x){
				return;
			}
			/*The function k_warp_reduce_sum expects shared memory size to be minimum two times the size
			of a warp*/
			constexpr int memsize=(THREADS_X_PER_BLOCK<=64)?64:THREADS_X_PER_BLOCK;
			static __shared__ F sdata[memsize];
			F* r_ptr=r;
			int tx=threadIdx.x;
			sdata[tx]=F(0.0);
			F partial_sum=0.0;
			for (int hs=0;hs<n;hs+=gridDim.x*2*blockDim.x){
					r_ptr=r+hs;
					int index=blockIdx.x*2*blockDim.x+tx;

					sdata[tx]=(((index+hs)<n)?r_ptr[index]:0.0)+(((index+hs+blockDim.x)<n)? r_ptr[index+blockDim.x]:0.0);
			
					__syncthreads();
					
					if (THREADS_X_PER_BLOCK>=512){
						if (tx<256){
						sdata[tx]+=sdata[tx+256];							
						}
						__syncthreads();
					}
					if (THREADS_X_PER_BLOCK>=256){
						if (tx<128){
							sdata[tx]+=sdata[tx+128];				
						}
						__syncthreads();
					}	
					if (THREADS_X_PER_BLOCK>=128){
						if (tx<64){
							sdata[tx]+=sdata[tx+64];				
						}
						__syncthreads();
					}
					
					if (tx<32){
						k_warp_reduce_sum<THREADS_X_PER_BLOCK>(sdata,tx);
						partial_sum+=sdata[0];
						sdata[tx]=0.0;
					}
					
					__syncthreads();	
			}
			//Because all threads write the same value, an if statement is not required
			if (tx==0){
				r[blockIdx.x*stride_r]=partial_sum;
			}
		}

		template<class F>
		__host__
		void reduce_sum(int n, F* r_d, int stride_r){
			if (n==0){
				return;
			}

			int threads_per_block=reductions::blocksizes::power_of_two_and_below_1024::MX;	
			while (n>1){
				int blocks=ceil(static_cast<F>(n)/(2*threads_per_block));
				
				switch(threads_per_block){
				
					case 2:{
						k_reduce_sum<2,F><<<blocks,2>>>(n, r_d,stride_r);
						break;
					}
					case 4:{
						k_reduce_sum<4,F><<<blocks,4>>>(n, r_d,stride_r);
						break;
					}
					case 8:{
						k_reduce_sum<8,F><<<blocks,8>>>(n, r_d,stride_r);
						break;
					}
					case 16:{
						k_reduce_sum<16,F><<<blocks,16>>>(n, r_d,stride_r);
						break;
					}
					case 32:{
						k_reduce_sum<32,F><<<blocks,32>>>(n, r_d,stride_r);
						break;
					}
					case 64:{
						k_reduce_sum<64,F><<<blocks,64>>>(n, r_d,stride_r);
						break;
					}
					case 128:{
						k_reduce_sum<128,F><<<blocks,128>>>(n, r_d,stride_r);
						break;
					}
					case 256:{
						k_reduce_sum<256,F><<<blocks,256>>>(n, r_d,stride_r);
						break;
					}		
					case 512:{
						k_reduce_sum<512,F><<<blocks,512>>>(n, r_d,stride_r);
						break;
					}			
				}		
				n=blocks;
			}
		}

		template<unsigned int THREADS_X_PER_BLOCK,unsigned int THREADS_Y_PER_BLOCK>
		__global__
		void k_reduce_sum_of_squares_poisson(float alpha_inv, float beta, int boundary_padding_thickness, int m, int k, cudaTextureObject_t A,cudaTextureObject_t B, float* r, int stride_r){
			m-=2*boundary_padding_thickness;
			k-=2*boundary_padding_thickness;
			if (k< (blockIdx.x*2*blockDim.x) || m<(blockIdx.y*blockDim.y)){
				return;
			}		
			
			int effective_gridDim_x=ceil(k/(2.0*THREADS_X_PER_BLOCK));
			constexpr int blocksize=THREADS_X_PER_BLOCK*THREADS_Y_PER_BLOCK;

			constexpr int memsize=(blocksize<=64)?64:blocksize;
			static __shared__ float sdata[memsize];
			
			float partial_sum=0;
			
			int idx=blockIdx.x*2*blockDim.x+threadIdx.x+boundary_padding_thickness;
			int idy=blockIdx.y*blockDim.y+threadIdx.y+boundary_padding_thickness;
			int tx=threadIdx.x;
			int ty=threadIdx.y;
			int index=ty*blockDim.x+tx;
			
			sdata[index]=0.0;

			for (int fy=idy;fy<=m;fy+=gridDim.y*blockDim.y){				
				for (int fx=idx;fx<=k;fx+=gridDim.x*2*blockDim.x){		
					float2 v=tex2D<float2>(A,fx+0.5,fy+0.5);
					float2 vlower=tex2D<float2>(A,fx+0.5,fy-1+0.5);
					float2 vupper=tex2D<float2>(A,fx+0.5,fy+1+0.5);
					float2 vleft=tex2D<float2>(A,fx-1+0.5,fy+0.5);
					float2 vright=tex2D<float2>(A,fx+1+0.5,fy+0.5);
					float2 b=tex2D<float2>(B,fx+0.5,fy+0.5);
					
					float2 diff;
					diff.x=b.x-(beta*v.x-vleft.x-vright.x-vupper.x-vlower.x)*alpha_inv;
					diff.y=b.y-(beta*v.y-vleft.y-vright.y-vupper.y-vlower.y)*alpha_inv;

					sdata[index]=diff.x*diff.x+diff.y*diff.y;

					if ((fx+blockDim.x)<=k && (fy<=m)){
						float2 v=tex2D<float2>(A,fx+blockDim.x+0.5,fy+0.5);
						float2 vlower=tex2D<float2>(A,fx+blockDim.x+0.5,fy-1+0.5);
						float2 vupper=tex2D<float2>(A,fx+blockDim.x+0.5,fy+1+0.5);
						float2 vleft=tex2D<float2>(A,fx+blockDim.x-1+0.5,fy+0.5);
						float2 vright=tex2D<float2>(A,fx+blockDim.x+1+0.5,fy+0.5);
						float2 b=tex2D<float2>(B,fx+blockDim.x+0.5,fy+0.5);
					
						float2 diff;
						diff.x=b.x-(beta*v.x-vleft.x-vright.x-vupper.x-vlower.x)*alpha_inv;
						diff.y=b.y-(beta*v.y-vleft.y-vright.y-vupper.y-vlower.y)*alpha_inv;				
			
						sdata[index]+=diff.x*diff.x+diff.y*diff.y;						
					}				
					__syncthreads();
					if (blocksize>=1024){
						if (index<512){
						sdata[index]+=sdata[index+512];							
						}
						__syncthreads();
					}
					
					if (blocksize>=512){
						if (index<256){
						sdata[index]+=sdata[index+256];							
						}
						__syncthreads();
					}
					if (blocksize>=256){
						if (index<128){
							sdata[index]+=sdata[index+128];				
						}
						__syncthreads();
					}	
					if (blocksize>=128){
						if (index<64){
							sdata[index]+=sdata[index+64];				
						}
						__syncthreads();
					}
					if (index<32){
						k_warp_reduce_sum<blocksize,float>(sdata,index);
						partial_sum+=sdata[0];
						sdata[index]=0;
					}
					__syncthreads();			
				}
			}
			if (tx==0 && ty==0){
				r[(blockIdx.y*effective_gridDim_x+blockIdx.x)*stride_r]=partial_sum;
			}
		}
		
		
		template<class F,class F2>
		__host__
		cudaError_t reduce_sum_of_squares_poisson(F alpha, F beta, int boundary_padding_thickness, int m, int k, cudaTextureObject_t A_tex, cudaTextureObject_t B_tex, F* r_d, int stride_r){
			if (r_d==nullptr || alpha==0.0 || beta ==0.0 ){
				if (r_d==nullptr){
					return cudaErrorInvalidValue;
				}
				else{
					return cudaSuccess;
				}
			}
			
			F alpha_inv=1.0/alpha;
			constexpr int threads_per_block_x=reductions::blocksizes::power_of_two_and_jointly_below_1024::MX;
			constexpr int threads_per_block_y=reductions::blocksizes::power_of_two_and_jointly_below_1024::MY;
			
			int blocks_x=ceil(static_cast<F>(k)/(2*threads_per_block_x));
			int blocks_y=ceil(static_cast<F>(m)/(2*threads_per_block_y));
			
			dim3 blocks=dim3(blocks_x,blocks_y,1);
			dim3 threads=dim3(threads_per_block_x,threads_per_block_y,1);
			k_reduce_sum_of_squares_poisson<threads_per_block_x,threads_per_block_y><<<blocks,threads>>>(alpha_inv,beta,boundary_padding_thickness,m,k,A_tex,B_tex, r_d,stride_r);
			
			int n=blocks_x*blocks_y;
			
			reduce_sum<F>(n,r_d,stride_r);
			return cudaSuccess;
		}

		__global__
		void k_restrict2h(int m, int k, float2* dest, int pitch_dest, float2* src, int pitch_src);
		
		__global__
		void k_restrict(float hy, float hx, int m, int k, float2* dest, int pitch_dest,cudaTextureObject_t src);

		template<class F, class F2>
		__host__
		cudaError_t restrict(int m, int k, int m_r, int k_r, F2* dest, int pitch_dest, F2* src, int pitch_src){
			int threads_per_block_x=reductions::blocksizes::group2::MX;	
			int threads_per_block_y=reductions::blocksizes::group2::MY;		
			int blocks_x=ceil(static_cast<float>(k_r)/(threads_per_block_x));
			int blocks_y=ceil(static_cast<float>(m_r)/(threads_per_block_y));
			
			dim3 threads=dim3(threads_per_block_x,threads_per_block_y,1);
			dim3 blocks=dim3(blocks_x,blocks_y,1);

			if (((m%2)!=0)&&((m_r%2)!=0)&&((k%2)!=0)&&((k_r%2)!=0)){		
				k_restrict2h<<<blocks,threads>>>(m_r,k_r, dest, pitch_dest, src,pitch_src);
			}
			else{

				//Create Resource descriptions
				cudaResourceDesc resDesc;
				memset(&resDesc,0,sizeof(resDesc));

				resDesc.resType = cudaResourceTypePitch2D;
				resDesc.res.pitch2D.devPtr=src;
				resDesc.res.pitch2D.width=k;
				resDesc.res.pitch2D.height=m;
				resDesc.res.pitch2D.pitchInBytes=pitch_src;
				resDesc.res.pitch2D.desc=cudaCreateChannelDesc<F2>(); 

				//Create Texture description
				cudaTextureDesc texDesc;
				memset(&texDesc,0,sizeof(texDesc));
				texDesc.normalizedCoords = false;
				texDesc.filterMode = cudaFilterModeLinear;
				texDesc.readMode=cudaReadModeElementType;
				texDesc.addressMode[0] = cudaAddressModeClamp;
				texDesc.addressMode[1] = cudaAddressModeClamp;

				cudaTextureObject_t src_tex;
				cudaError_t err=cudaCreateTextureObject(&src_tex, &resDesc, &texDesc, NULL);
				
				if (err !=cudaSuccess){
					return err;
				}		
				F hx=static_cast<float>(k)/(k_r-1);
				F hy=static_cast<float>(m)/(m_r-1);
				k_restrict<<<blocks,threads>>>(hy, hx, m_r,k_r,dest, pitch_dest, src_tex);
			}
			return cudaSuccess;		
		}
		
		
		inline int restrict_n(int n){
			if ((n%2) != 0){
				return (n-1)*0.5+1;
			}
			else{ 
				return 0.5*(n-2)+2;
			}			
		}

		template<class F, class F2>
		cudaError_t restrict(int m, int k, int m_r, int k_r, F2* dest, int pitch_dest, F2* src, int pitch_src);
	}
}