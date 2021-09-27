#include<stdio.h>
#include "reductions.h"

namespace desal{
	namespace cuda{
		template<unsigned int THREADS_X_PER_BLOCK, class F>
		__device__
		void d_warp_reduce_sum_2D(volatile F* sdata, int ty, int tx){
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
		void d_warp_reduce_sum(volatile F* sdata, int tx){

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

		//AX=B
		template<unsigned int THREADS_X_PER_BLOCK,unsigned int THREADS_Y_PER_BLOCK, class F, class F2>
		__global__
		void k_reduce_sum_of_squares_poisson_field_residual(F alpha_inv, F beta, int boundary_padding_thickness, int n, cudaTextureObject_t A,cudaTextureObject_t B, F* r, int stride_r){
		//printf("Durch\n");
			n-=2*boundary_padding_thickness;
			if (n< (blockIdx.x*2*blockDim.x) || n<(blockIdx.y*blockDim.y)){
				return;
			}		
			
			int effective_gridDim_x=ceil(n/(2.0*THREADS_X_PER_BLOCK));
			//printf("n: %d, idx:%d, idy: %d\n",n,blockIdx.x*2*blockDim.x,blockIdx.y*blockDim.y);
			constexpr int blocksize=THREADS_X_PER_BLOCK*THREADS_Y_PER_BLOCK;

			constexpr int memsize=(blocksize<=64)?64:blocksize;
			static __shared__ F sdata[memsize];
			
			F partial_sum=0;
			
			int idy=blockIdx.y*blockDim.y+threadIdx.y;
			int idx=blockIdx.x*2*blockDim.x+threadIdx.x;
			int tx=threadIdx.x;
			int ty=threadIdx.y;
			int index=ty*blockDim.x+tx;
			
			sdata[index]=F(0.0); //initialize relevant part of the sdata array
			//printf("alpha_inv: %f, beta: %f\n",alpha_inv,beta);

			for (int fy=idy;fy<n;fy+=gridDim.y*blockDim.y){
				//int fy=idy+hs;
				//int fx=idx;		
				
				for (int fx=idx;fx<n;fx+=gridDim.x*2*blockDim.x){
					//fx+=ws;
			
					F2 v=tex2D<F2>(A,fx+boundary_padding_thickness+0.5,fy+boundary_padding_thickness+0.5);
					F2 vlower=tex2D<F2>(A,fx+boundary_padding_thickness+0.5,fy-1+boundary_padding_thickness+0.5);
					F2 vupper=tex2D<F2>(A,fx+boundary_padding_thickness+0.5,fy+1+boundary_padding_thickness+0.5);
					F2 vleft=tex2D<F2>(A,fx-1+boundary_padding_thickness+0.5,fy+boundary_padding_thickness+0.5);
					F2 vright=tex2D<F2>(A,fx+1+boundary_padding_thickness+0.5,fy+boundary_padding_thickness+0.5);
					
					F2 b=tex2D<F2>(B,fx+boundary_padding_thickness+0.5,fy+boundary_padding_thickness+0.5);
					
					F2 diff;
					diff.x=b.x-(beta*v.x-vleft.x-vright.x-vupper.x-vlower.x)*alpha_inv;
					diff.y=b.y-(beta*v.y-vleft.y-vright.y-vupper.y-vlower.y)*alpha_inv;
					sdata[index]=diff.x*diff.x+diff.y*diff.y;
					//printf("n:%d, vx:%f b.x:%f, u. %f %f %f %f diff:%f\n",n,v.y,b.y,vleft.y,vright.y,vlower.y,vupper.y,diff.x);
					//printf("sdata[index]=%f, vs: %f %f %f %f %f blockids: %d , %d\n",diff.x,v.x,vlower.x,vupper.x,vleft.x,vright.x,fy,fx);
					
					//printf("%f %f %f %f %f\n",v.x,vleft.x,vright.x,vupper.x,vlower.x);
					//printf("sdata %f\n",sdata[index]);
					//printf("y,x: %d, %d , diffx: %f diffy: %f\n",fy,fx,diff.x, diff.x);
			
					if ((fx+blockDim.x)<n && (fy)<n){
						F2 v=tex2D<F2>(A,fx+blockDim.x+boundary_padding_thickness+0.5,fy+boundary_padding_thickness+0.5);
						F2 vlower=tex2D<F2>(A,fx+blockDim.x+boundary_padding_thickness+0.5,fy-1+boundary_padding_thickness+0.5);
						F2 vupper=tex2D<F2>(A,fx+blockDim.x+boundary_padding_thickness+0.5,fy+1+boundary_padding_thickness+0.5);
						F2 vleft=tex2D<F2>(A,fx+blockDim.x-1+boundary_padding_thickness+0.5,fy+boundary_padding_thickness+0.5);
						F2 vright=tex2D<F2>(A,fx+blockDim.x+1+boundary_padding_thickness+0.5,fy+boundary_padding_thickness+0.5);

						F2 b=tex2D<F2>(B,fx+blockDim.x+boundary_padding_thickness+0.5,fy+boundary_padding_thickness+0.5);
					
						F2 diff;
						diff.x=b.x-(beta*v.x-vleft.x-vright.x-vupper.x-vlower.x)*alpha_inv;
						diff.y=b.y-(beta*v.y-vleft.y-vright.y-vupper.y-vlower.y)*alpha_inv;				
					
						//printf("sdata[index]=%f, vs: %f %f %f %f %f fy,fy: %d , %d\n",diff.x,v.x,vlower.x,vupper.x,vleft.x,vright.x,fy,fx+blockDim.x);
						sdata[index]+=diff.x*diff.x+diff.y*diff.y; //TODO: Add second element
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
					/*
					if (threadIdx.x==0 && threadIdx.y==0 && blockIdx.x==0 && blockIdx.y==0){
						for (int i=0;i<blocksize;i++){
						
						printf("%f,",sdata[i]);
						}
						printf("\n");
					}
					*/
					if (index<32){
						d_warp_reduce_sum<blocksize,F>(sdata,index);

						partial_sum+=sdata[0];
						

					}
					__syncthreads();			
				}
			}
			if (tx==0 && ty==0){
		//	printf("Partial sum: %f\n",partial_sum);
				r[(blockIdx.y*effective_gridDim_x+blockIdx.x)*stride_r]=partial_sum;
			}

		}


		/*Reduces an array with n elements to log_b(n) its size by summing its entries, whereas b is the number of blocks in the grid.*/
		template<unsigned int THREADS_X_PER_BLOCK, class F>
		__global__
		void k_reduce_sum(int n, F* r, int stride_r){
			if (n<blockIdx.x*2*blockDim.x){
				return;
			}
			/*The function d_warp_reduce_sum expects shared memory size to be minimum two times the size
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
						d_warp_reduce_sum<THREADS_X_PER_BLOCK>(sdata,tx);
						partial_sum+=sdata[0];
					//	printf("Add%f\n",sdata[0]);
						
					}
					
					__syncthreads();	
			}
			//Because all threads write the same value, the if statement is not required
			if (tx==0){
				r[blockIdx.x*stride_r]=partial_sum;
			}
		}

		template<class F>
		__host__
		void reduce_sum_device(int n, F* r_d, int stride_r){

			if (n==0){
				return;
			}

			int threads_per_block=2;	
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

		void reduce_sum_f32_device(int n, float* r_d, int stride_r){
			reduce_sum_device<float>(n, r_d, stride_r);
		}

		void reduce_sum_f64_device(int n, double* r_d, int stride_r){
			reduce_sum_device<double>(n, r_d, stride_r);
		}


		template<class F, class F2>
		__host__
		void reduce_sum_of_squares_poisson_field_residual_device(F alpha, F beta, int boundary_padding_thickness, int n, F2* A_d,int pitch_a, F2* B_d, int pitch_b, F* r_d, int stride_r){
			if (r_d==nullptr || alpha==0.0 || beta ==0.0 ){
				return;
			}
			
			F alpha_inv=1.0/alpha;
			constexpr int threads_per_block_x=512;
			constexpr int threads_per_block_y=2;
			
			//TODO: Check if both variables above are power of two and smaller than 1024
				
			//Create Resource descriptions
			cudaResourceDesc resDescA;
			memset(&resDescA,0,sizeof(resDescA));
			resDescA.resType = cudaResourceTypePitch2D;
			resDescA.res.pitch2D.devPtr=A_d;
			resDescA.res.pitch2D.width=n-boundary_padding_thickness;
			resDescA.res.pitch2D.height=n-boundary_padding_thickness;
			resDescA.res.pitch2D.pitchInBytes=pitch_a;
			resDescA.res.pitch2D.desc=cudaCreateChannelDesc<F2>(); //is equivalent to cudaCreateChannelDesc(32,32,0,0,cudaChannelFormatKindFloat)

			cudaResourceDesc resDescB;
			memset(&resDescB,0,sizeof(resDescB));
			resDescB.resType = cudaResourceTypePitch2D;
			resDescB.res.pitch2D.devPtr=B_d;
			resDescB.res.pitch2D.width=n-boundary_padding_thickness;
			resDescB.res.pitch2D.height=n-boundary_padding_thickness;
			resDescB.res.pitch2D.pitchInBytes=pitch_b;
			resDescB.res.pitch2D.desc=cudaCreateChannelDesc<F2>(); //is equivalent to cudaCreateChannelDesc(32,32,0,0,cudaChannelFormatKindFloat)


			//Create Texture description
			cudaTextureDesc texDesc;
			memset(&texDesc,0,sizeof(texDesc));
			texDesc.normalizedCoords = false;
			texDesc.filterMode = cudaFilterModePoint;
			texDesc.readMode=cudaReadModeElementType;
			texDesc.addressMode[0] = cudaAddressModeBorder;
			texDesc.addressMode[1] = cudaAddressModeBorder;

			//Create Texture Object
			cudaTextureObject_t A_tex;
			cudaTextureObject_t B_tex;
			//printf("nOn: %d\n",n*n);
			cudaError_t error1=cudaCreateTextureObject(&A_tex, &resDescA, &texDesc, NULL);
			cudaError_t error2=cudaCreateTextureObject(&B_tex, &resDescB, &texDesc, NULL);
			if ((error1 !=cudaSuccess)&&(error2 !=cudaSuccess)){
				printf("Errorcode: %d\n",error1);
			}
				
			int blocks_x=ceil(static_cast<F>(n)/(2*threads_per_block_x));
			int blocks_y=ceil(static_cast<F>(n)/(2*threads_per_block_y));
			
			dim3 blocks=dim3(blocks_x,blocks_y,1);
			dim3 threads=dim3(threads_per_block_x,threads_per_block_y,1);
			k_reduce_sum_of_squares_poisson_field_residual<threads_per_block_x,threads_per_block_y,F,F2><<<blocks,threads>>>(alpha_inv,beta,boundary_padding_thickness,n, A_tex,B_tex, r_d,stride_r);
			
			n=blocks_x*blocks_y;
			
			reduce_sum_f32_device(n,r_d,stride_r);
			
		}


		template<class F, class F2>
		__host__
		void reduce_sum_of_squares_poisson_field_residual_device(F alpha, F beta, int boundary_padding_thickness, int n, F2* A_d,int pitch_a, cudaTextureObject_t B_tex, F* r_d, int stride_r){
			if (r_d==nullptr || alpha==0.0 || beta ==0.0 ){
				return;
			}
			
			F alpha_inv=1.0/alpha;
			constexpr int threads_per_block_x=512;
			constexpr int threads_per_block_y=2;
			
			//TODO: Check if both variables above are power of two and smaller than 1024
				
			//Create Resource descriptions
			cudaResourceDesc resDescA;
			memset(&resDescA,0,sizeof(resDescA));
			resDescA.resType = cudaResourceTypePitch2D;
			resDescA.res.pitch2D.devPtr=A_d;
			resDescA.res.pitch2D.width=n-boundary_padding_thickness;
			resDescA.res.pitch2D.height=n-boundary_padding_thickness;
			resDescA.res.pitch2D.pitchInBytes=pitch_a;
			resDescA.res.pitch2D.desc=cudaCreateChannelDesc<F2>(); //is equivalent to cudaCreateChannelDesc(32,32,0,0,cudaChannelFormatKindFloat)

			//Create Texture description
			cudaTextureDesc texDesc;
			memset(&texDesc,0,sizeof(texDesc));
			texDesc.normalizedCoords = false;
			texDesc.filterMode = cudaFilterModePoint;
			texDesc.readMode=cudaReadModeElementType;
			texDesc.addressMode[0] = cudaAddressModeBorder;
			texDesc.addressMode[1] = cudaAddressModeBorder;

			//Create Texture Object
			cudaTextureObject_t A_tex;
			//printf("nOn: %d\n",n*n);
			cudaError_t error1=cudaCreateTextureObject(&A_tex, &resDescA, &texDesc, NULL);
			if ((error1 !=cudaSuccess)){
				printf("Errorcode: %d\n",error1);
			}
				
			int blocks_x=ceil(static_cast<F>(n)/(2*threads_per_block_x));
			int blocks_y=ceil(static_cast<F>(n)/(2*threads_per_block_y));
			
			dim3 blocks=dim3(blocks_x,blocks_y,1);
			dim3 threads=dim3(threads_per_block_x,threads_per_block_y,1);
			k_reduce_sum_of_squares_poisson_field_residual<threads_per_block_x,threads_per_block_y,F,F2><<<blocks,threads>>>(alpha_inv,beta,boundary_padding_thickness,n, A_tex,B_tex, r_d,stride_r);
			
			n=blocks_x*blocks_y;
			
			reduce_sum_f32_device(n,r_d,stride_r);
		}

		__host__
		void reduce_sum_of_squares_poisson_field_residual_f32_device(float alpha, float beta, int boundary_padding_thickness, int n, float2* A_d,int pitch_a, float2* B_d, int pitch_b, float* r_d, int stride_r){
			reduce_sum_of_squares_poisson_field_residual_device<float,float2>(alpha, beta, boundary_padding_thickness, n, A_d,pitch_a, B_d, pitch_b, r_d, stride_r);
		}

		template<class F2>
		__global__
		void k_restrict2h(int n, F2* dest, int pitch_dest, F2* src, int pitch_src){
			int idy=blockIdx.y*blockDim.y+threadIdx.y;
			int idx=blockIdx.x*blockDim.x+threadIdx.x;
			
			dest=(F2*) ((char*)dest+idy*pitch_dest);
			src=(F2*) ((char*)src+2*idy*pitch_src);
			
			for(int i=idy;i<n;i+=gridDim.y*blockDim.y){
				for(int j = idx; j<n; j+=gridDim.x*blockDim.x){
					dest[j].x=src[2*j].x;
					dest[j].y=src[2*j].y;													
				}
				dest=(F2*) ((char*)dest+pitch_dest);
				src=(F2*) ((char*)src+2*pitch_src);	
			}
		}

		template<class F, class F2>
		__global__
		void k_restrict(F h, int n, F2* dest, int pitch_dest,cudaTextureObject_t src){
			int idy=blockIdx.y*blockDim.y+threadIdx.y;
			int idx=blockIdx.x*blockDim.x+threadIdx.x;
			
			dest=(F2*) ((char*)dest+idy*pitch_dest);
			
			for(int i=idy;i<n;i+=gridDim.y*blockDim.y){
				for(int j = idx; j<n; j+=gridDim.x*blockDim.x){
					F2 v=tex2D<F2>(src,h*j+0.5,h*i+0.5);
					//printf("i: %d j: %d val: %f\n",i,j,v.x);
					dest[j].x+=v.x;
					dest[j].y+=v.y;													
				}
				dest=(F2*) ((char*)dest+pitch_dest);
			}
		}

		template<class F, class F2>
		__host__
		void restrict(int n, int n_r, F2* dest, int pitch_dest, F2* src, int pitch_src){
			int threads_per_block_x=8;	
			int threads_per_block_y=4;	
			int blocks_x=ceil(static_cast<float>(n_r)/(threads_per_block_x));
			int blocks_y=ceil(static_cast<float>(n_r)/(threads_per_block_y));
			
			dim3 threads=dim3(threads_per_block_x,threads_per_block_y,1);
			dim3 blocks=dim3(blocks_x,blocks_y,1);
			//printf("n: %d, nr: %d\n",n,n_r);
			if (((n%2)==0)&&((n_r%2)==0)){		
				k_restrict2h<F2><<<blocks,threads>>>(n_r,dest, pitch_dest, src,pitch_src);
			}
			else{

				//Create Resource descriptions
				cudaResourceDesc resDesc;
				memset(&resDesc,0,sizeof(resDesc));

				resDesc.resType = cudaResourceTypePitch2D;
				resDesc.res.pitch2D.devPtr=src;
				resDesc.res.pitch2D.width=n;
				resDesc.res.pitch2D.height=n;
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
				//printf("nOn: %d\n",n*n);
				cudaError_t error1=cudaCreateTextureObject(&src_tex, &resDesc, &texDesc, NULL);
				
				if ((error1 !=cudaSuccess)){
					printf("Errorcode prolongand add: %d\n",error1);
				}		
				F h=static_cast<F>(n)/(n_r-1);
				k_restrict<F,F2><<<blocks,threads>>>(h,n_r,dest, pitch_dest, src_tex);
			}
			
		}

		template
		void restrict<float, float2>(int n, int n_r, float2* dest, int pitch_dest, float2* src, int pitch_src);

		/*
		template
		void restrict<double, double2>(int n, int n_r,  double2* dest, int pitch_dest, double2* src, int pitch_src);

		*/
	}
}