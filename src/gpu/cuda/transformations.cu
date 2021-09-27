//AX=B
#include <stdio.h>

#include "transformations.h"

namespace desal{
			namespace cuda{
		template< class F, class F2>
		__global__
		void k_transform_entries_into_residuals(F alpha_inv, F beta, int boundary_padding_thickness, int m, int k, cudaTextureObject_t A,cudaTextureObject_t B, F2* r, int pitch_r){
		//printf("Durch\n");
			m-=2*boundary_padding_thickness;
			k-=2*boundary_padding_thickness;
			if (k< (blockIdx.x*blockDim.x) || m<(blockIdx.y*blockDim.y)){
				return;
			}
			//printf("n: %d, idx:%d, idy: %d\n",n,blockIdx.x*blockDim.x,blockIdx.y*blockDim.y);

			int idy=blockIdx.y*blockDim.y+threadIdx.y;
			int idx=blockIdx.x*blockDim.x+threadIdx.x;
			int tx=threadIdx.x;
			int ty=threadIdx.y;
			F2* r_ptr=(F2*) ((char*)r+(idy+boundary_padding_thickness)*pitch_r);
			
			for (int fy=idy;fy<m;fy+=gridDim.y*blockDim.y){	
					
				for (int fx=idx;fx<k;fx+=gridDim.x*blockDim.x){			
					F2 v=tex2D<F2>(A,fx+boundary_padding_thickness+0.5,fy+boundary_padding_thickness+0.5);
					F2 vlower=tex2D<F2>(A,fx+boundary_padding_thickness+0.5,fy-1+boundary_padding_thickness+0.5);
					F2 vupper=tex2D<F2>(A,fx+boundary_padding_thickness+0.5,fy+1+boundary_padding_thickness+0.5);
					F2 vleft=tex2D<F2>(A,fx-1+boundary_padding_thickness+0.5,fy+boundary_padding_thickness+0.5);
					F2 vright=tex2D<F2>(A,fx+1+boundary_padding_thickness+0.5,fy+boundary_padding_thickness+0.5);			
					
					F2 b=tex2D<F2>(B,fx+boundary_padding_thickness+0.5,fy+boundary_padding_thickness+0.5);
					F2 diff;
					diff.x=b.x-(beta*v.x-vleft.x-vright.x-vupper.x-vlower.x)*alpha_inv;
					diff.y=b.y-(beta*v.y-vleft.y-vright.y-vupper.y-vlower.y)*alpha_inv;
					
					r_ptr[fx+boundary_padding_thickness].x=diff.x;
					r_ptr[fx+boundary_padding_thickness].y=diff.y;
				}		
				r_ptr=(F2*) ((char*)r_ptr+pitch_r);	
			}
		}

		template<class F, class F2>
		__host__
		void transform_entries_into_residuals_device(F alpha, F beta, int boundary_padding_thickness, int m, int k, F2* A_d, int pitch_a, F2* B_d, int pitch_b, F2* r_d, int pitch_r){
			F alpha_inv=1.0/alpha;
			
			//Create Resource descriptions
			cudaResourceDesc resDescA;
			memset(&resDescA,0,sizeof(resDescA));
			resDescA.resType = cudaResourceTypePitch2D;
			resDescA.res.pitch2D.devPtr=A_d;
			resDescA.res.pitch2D.width=k;
			resDescA.res.pitch2D.height=m;
			resDescA.res.pitch2D.pitchInBytes=pitch_a;
			resDescA.res.pitch2D.desc=cudaCreateChannelDesc<F2>(); 

			cudaResourceDesc resDescB;
			memset(&resDescB,0,sizeof(resDescB));
			resDescB.resType = cudaResourceTypePitch2D;
			resDescB.res.pitch2D.devPtr=B_d;
			resDescB.res.pitch2D.width=k;
			resDescB.res.pitch2D.height=m;
			resDescB.res.pitch2D.pitchInBytes=pitch_b;
			resDescB.res.pitch2D.desc=cudaCreateChannelDesc<F2>(); 

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
			//	printf("Errorcode: %d\n",error1);
			}
			
			int threads_per_block_x=8;	
			int threads_per_block_y=4;	
			int blocks_x=ceil(static_cast<F>(k)/(threads_per_block_x));
			int blocks_y=ceil(static_cast<F>(m)/(threads_per_block_y));
			
			dim3 threads=dim3(threads_per_block_x,threads_per_block_y,1);
			dim3 blocks=dim3(blocks_x,blocks_y,1);
			
			k_transform_entries_into_residuals<F,F2><<<blocks,threads>>>(alpha_inv, beta, boundary_padding_thickness, m, k,A_tex,B_tex, r_d, pitch_r);	
				
		}

		__host__
		void transform_entries_into_residuals_f32_device(float alpha, float beta, int boundary_padding_thickness, int m, int k, float2* A_d, int pitch_a, float2* B_d, int pitch_b, float2* r_d, int pitch_r){

			transform_entries_into_residuals_device<float,float2>(alpha, beta, boundary_padding_thickness, m, k, A_d, pitch_a, B_d, pitch_b, r_d, pitch_r);

		}

		//Prolongs source 2D array and adds it to destination
		template<class F2>
		__global__
		void k_prolong_by_interpolation_and_add_2h(int m, int k, F2* dest, int pitch_dest, cudaTextureObject_t src){
			int idy=blockIdx.y*blockDim.y+threadIdx.y;
			int idx=blockIdx.x*blockDim.x+threadIdx.x;
			
			dest=(F2*) ((char*)dest+idy*pitch_dest);

			for(int i=idy;i<m;i+=gridDim.y*blockDim.y){
				for(int j = idx; j<k; j+=gridDim.x*blockDim.x){
					F2 v=tex2D<F2>(src,0.5*j+0.5,0.5*i+0.5);
					dest[j].x+=v.x;
					dest[j].y+=v.y;													
				}
				dest=(F2*) ((char*)dest+pitch_dest);
			}	
		}

		//Prolongs source 2D array and adds it to destination
		template<class F, class F2>
		__global__
		void k_prolong_by_interpolation_and_add(F h, int n, F2* dest, int pitch_dest, cudaTextureObject_t src){
			int idy=blockIdx.y*blockDim.y+threadIdx.y;
			int idx=blockIdx.x*blockDim.x+threadIdx.x;
			
			dest=(F2*) ((char*)dest+idy*pitch_dest);

			for(int i=idy;i<n;i+=gridDim.y*blockDim.y){
				for(int j = idx; j<n; j+=gridDim.x*blockDim.x){
					F2 v=tex2D<F2>(src,h*j+0.5,h*i+0.5);
					dest[j].x+=v.x;
					dest[j].y+=v.y;													
				}
				dest=(F2*) ((char*)dest+pitch_dest);
			}	
		}

		//Prolongs source 2D array and adds it to destination
		template<class F2>
		__host__
		void prolong_by_interpolation_and_add_2h(int m, int k, F2* dest, int pitch_dest, F2* src, int pitch_src){

			//Create Resource descriptions
			cudaResourceDesc resDesc;
			memset(&resDesc,0,sizeof(resDesc));

			resDesc.resType = cudaResourceTypePitch2D;
			resDesc.res.pitch2D.devPtr=src;
			resDesc.res.pitch2D.width=ceil((k-1)*0.5+1);
			resDesc.res.pitch2D.height=ceil((m-1)*0.5+1);
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
			//printf("nOn: %d\n",n*n);
			cudaError_t error1=cudaCreateTextureObject(&src_tex, &resDesc, &texDesc, NULL);
			
			if ((error1 !=cudaSuccess)){
				printf("Errorcode prolongand add: %s\n",cudaGetErrorString(error1));
				exit(1);
			}	
			int threads_per_block_x=8;	
			int threads_per_block_y=4;	
			int blocks_x=ceil(static_cast<float>(m)/(threads_per_block_x));
			int blocks_y=ceil(static_cast<float>(k)/(threads_per_block_y));
			
			dim3 threads=dim3(threads_per_block_x,threads_per_block_y,1);
			dim3 blocks=dim3(blocks_x,blocks_y,1);
			k_prolong_by_interpolation_and_add_2h<F2><<<blocks,threads>>>(m,k, dest, pitch_dest, src_tex);
		}

		//Prolongs source 2D array and adds it to destination
		template<class F, class F2>
		__host__
		void prolong_by_interpolation_and_add(F h, int n_p, int n_r, F2* dest, int pitch_dest, F2* src, int pitch_src){

			//Create Resource descriptions
			cudaResourceDesc resDesc;
			memset(&resDesc,0,sizeof(resDesc));

			resDesc.resType = cudaResourceTypePitch2D;
			resDesc.res.pitch2D.devPtr=src;
			resDesc.res.pitch2D.width=n_r;
			resDesc.res.pitch2D.height=n_r;
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
			cudaError_t error1=cudaCreateTextureObject(&src_tex, &resDesc, &texDesc, NULL);
			
			if ((error1 !=cudaSuccess)){
				printf("Errorcode prolongand add: %s\n",cudaGetErrorString(error1));
				exit(1);
			}	
			int threads_per_block_x=8;	
			int threads_per_block_y=4;	
			int blocks_x=ceil(static_cast<float>(n_p)/(threads_per_block_x));
			int blocks_y=ceil(static_cast<float>(n_p)/(threads_per_block_y));
			
			dim3 threads=dim3(threads_per_block_x,threads_per_block_y,1);
			dim3 blocks=dim3(blocks_x,blocks_y,1);
			k_prolong_by_interpolation_and_add<F,F2><<<blocks,threads>>>(h,n_p, dest, pitch_dest, src_tex);
		}

		//template<class F2>
		__host__
		void prolong_and_add(int n_p,int n_r, float2* dest, int pitch_dest, float2* src, int pitch_src){
			using F2=float2;

			if (((n_p)%2)!=0){
				prolong_by_interpolation_and_add_2h<F2>(n_r, n_r, dest, pitch_dest,src,pitch_src);	
			}
			else{
				float h=static_cast<float>(n_r)/(n_p-1);
				printf("Hier: %f  %d %d\n",h, n_r, n_p);
				prolong_by_interpolation_and_add<float,F2>(h,n_p, n_r, dest, pitch_dest,src,pitch_src);	
			}
			
		}

		__host__
		void prolong_and_add(int n_p, int n_r, float2* dest, int pitch_dest, float2* src, int pitch_src);
	}
}
