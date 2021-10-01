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
		cudaError_t transform_entries_into_residuals_device(F alpha, F beta, int boundary_padding_thickness, int m, int k, F2* A_d, int pitch_a, F2* B_d, int pitch_b, F2* r_d, int pitch_r){
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
				return (error1!=cudaSuccess)?error1:error2;
			}
			
			int threads_per_block_x=8;	
			int threads_per_block_y=4;	
			int blocks_x=ceil(static_cast<F>(k)/(threads_per_block_x));
			int blocks_y=ceil(static_cast<F>(m)/(threads_per_block_y));
			
			dim3 threads=dim3(threads_per_block_x,threads_per_block_y,1);
			dim3 blocks=dim3(blocks_x,blocks_y,1);
			
			k_transform_entries_into_residuals<F,F2><<<blocks,threads>>>(alpha_inv, beta, boundary_padding_thickness, m, k,A_tex,B_tex, r_d, pitch_r);	
			return cudaSuccess;
		}

		template
		__host__
		cudaError_t transform_entries_into_residuals_device(float alpha, float beta, int boundary_padding_thickness, int m, int k, float2* A_d, int pitch_a, float2* B_d, int pitch_b, float2* r_d, int pitch_r);

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
		void k_prolong_by_interpolation_and_add(F hy, F hx, int m, int k, F2* dest, int pitch_dest, cudaTextureObject_t src){
			int idy=blockIdx.y*blockDim.y+threadIdx.y;
			int idx=blockIdx.x*blockDim.x+threadIdx.x;
			
			dest=(F2*) ((char*)dest+idy*pitch_dest);

			for(int i=idy;i<m;i+=gridDim.y*blockDim.y){
				for(int j = idx; j<k; j+=gridDim.x*blockDim.x){
					F2 v=tex2D<F2>(src,hx*j+0.5,hy*i+0.5);
					dest[j].x+=v.x;
					dest[j].y+=v.y;													
				}
				dest=(F2*) ((char*)dest+pitch_dest);
			}	
		}

		//Prolongs source 2D array and adds it to destination
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
			//printf("nOn: %d\n",n*n);
			
			cudaError_t err=cudaCreateTextureObject(&src_tex, &resDesc, &texDesc, NULL);
			
			if (err !=cudaSuccess){
				return err;
			}
			int threads_per_block_x=256;	
			int threads_per_block_y=4;	
			int blocks_x=ceil(static_cast<float>(k_p)/(threads_per_block_x));
			int blocks_y=ceil(static_cast<float>(m_p)/(threads_per_block_y));
			
			dim3 threads=dim3(threads_per_block_x,threads_per_block_y,1);
			dim3 blocks=dim3(blocks_x,blocks_y,1);
			k_prolong_by_interpolation_and_add_2h<F2><<<blocks,threads>>>(m_p,k_p, dest, pitch_dest, src_tex);
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
			int threads_per_block_x=256;	
			int threads_per_block_y=4;	
			int blocks_x=ceil(static_cast<float>(k_p)/(threads_per_block_x));
			int blocks_y=ceil(static_cast<float>(m_p)/(threads_per_block_y));
			
			dim3 threads=dim3(threads_per_block_x,threads_per_block_y,1);
			dim3 blocks=dim3(blocks_x,blocks_y,1);
			k_prolong_by_interpolation_and_add<F,F2><<<blocks,threads>>>(hy,hx,m_p,k_p, dest, pitch_dest, src_tex);
			return cudaSuccess;
		}

		//template<class F2>
		__host__
		cudaError_t prolong_and_add(int m_p, int k_p, int m_r, int k_r, float2* dest, int pitch_dest, float2* src, int pitch_src){
			using F2=float2;

			if ((((m_p)%2)!=0)&&((k_p%2)!=0)){
				return prolong_by_interpolation_and_add_2h<F2>(m_p, k_p, m_r, k_r, dest, pitch_dest,src,pitch_src);	
			}
			else{
				float hy=static_cast<float>(m_r)/(m_p-1);
				float hx=static_cast<float>(k_r)/(k_p-1);
				//printf("Hier: %f  %d %d\n",h, n_r, n_p);
				return prolong_by_interpolation_and_add<float,F2>(hy,hx,m_p,k_p,m_r,k_r, dest, pitch_dest,src,pitch_src);	
			}
			
		}
		
		template<class F, class F2>
		__global__
		void k_divergence(F one_half_dy, F one_half_dx, int boundary_padding_thickness, int m, int k,cudaTextureObject_t A, F2* C, size_t pitch_c){
			m-=2*boundary_padding_thickness;
			k-=2*boundary_padding_thickness;
			if (k< (blockIdx.x*blockDim.x) || m<(blockIdx.y*blockDim.y)){
				return;
			}
			
			int idy=blockIdx.y*blockDim.y+threadIdx.y;
			int idx=blockIdx.x*blockDim.x+threadIdx.x;
			C=((F2*) ((char*)C+(idy+boundary_padding_thickness)*pitch_c))+boundary_padding_thickness;
		
			for (int i=idy;i<m;i+=gridDim.y*blockDim.y){	
					
				for (int j=idx;j<k;j+=gridDim.x*blockDim.x){				
					F2 v=tex2D<F2>(A,i+boundary_padding_thickness+0.5,j+boundary_padding_thickness+0.5);
					F2 vlower=tex2D<F2>(A,i+boundary_padding_thickness+0.5,j-1+boundary_padding_thickness+0.5);
					F2 vupper=tex2D<F2>(A,i+boundary_padding_thickness+0.5,j+1+boundary_padding_thickness+0.5);
					F2 vleft=tex2D<F2>(A,i-1+boundary_padding_thickness+0.5,j+boundary_padding_thickness+0.5);
					F2 vright=tex2D<F2>(A,i+1+boundary_padding_thickness+0.5,j+boundary_padding_thickness+0.5);			
					C[j].x=one_half_dx*(vleft.x-vright.x)+one_half_dy*(+vupper.y-vlower.y);
					C[j].y=0;
				}		
				C=(F2*) ((char*)C+gridDim.y*blockDim.y*pitch_c);	
			}		
		
		}
	

		template<class F, class F2>
		cudaError_t divergence(F dy, F dx, int boundary_padding_thickness, int m, int k,F2* A_d, size_t pitch_a, F2* C_d, size_t pitch_c){
			//Create Resource descriptions
			cudaResourceDesc resDesc;
			memset(&resDesc,0,sizeof(resDesc));
			resDesc.resType = cudaResourceTypePitch2D;
			resDesc.res.pitch2D.devPtr=A_d;
			resDesc.res.pitch2D.width=k;
			resDesc.res.pitch2D.height=m;
			resDesc.res.pitch2D.pitchInBytes=pitch_a;
			resDesc.res.pitch2D.desc=cudaCreateChannelDesc<F2>(); 

			//Create Texture description
			cudaTextureDesc texDesc;
			memset(&texDesc,0,sizeof(texDesc));
			texDesc.normalizedCoords = false;
			texDesc.filterMode = cudaFilterModeLinear;
			texDesc.readMode=cudaReadModeElementType;
			texDesc.addressMode[0] = cudaAddressModeBorder;
			texDesc.addressMode[1] = cudaAddressModeBorder;

			cudaTextureObject_t A_tex;
			cudaError_t err=cudaCreateTextureObject(&A_tex, &resDesc, &texDesc, NULL);
			
			if (err !=cudaSuccess){
				return err;
			}				
			int threads_per_block_x=256;	
			int threads_per_block_y=4;	
			int blocks_x=ceil(static_cast<float>(k)/(threads_per_block_x));
			int blocks_y=ceil(static_cast<float>(m)/(threads_per_block_y));
			
			dim3 threads=dim3(threads_per_block_x,threads_per_block_y,1);
			dim3 blocks=dim3(blocks_x,blocks_y,1);
			k_divergence<F,F2><<<blocks,threads>>>(0.5*dy, 0.5*dx, boundary_padding_thickness, m, k,A_tex, C_d, pitch_c);		
			
			return cudaSuccess;
			
		}
		
		template
		cudaError_t divergence(float dy, float dx, int boundary_padding_thickness, int m, int k, float2* A_d, size_t pitch_a, float2* C_d, size_t pitch_c);

		template<class F, class F2>
		__global__
		void k_subtract_gradient(F one_half_dy, F one_half_dx, int boundary_padding_thickness, int m, int k,cudaTextureObject_t A, F2* C, size_t pitch_c){
			m-=2*boundary_padding_thickness;
			k-=2*boundary_padding_thickness;
			if (k< (blockIdx.x*blockDim.x) || m<(blockIdx.y*blockDim.y)){
				return;
			}
			
			int idy=blockIdx.y*blockDim.y+threadIdx.y;
			int idx=blockIdx.x*blockDim.x+threadIdx.x;
			C=(F2*) ((char*)C+(idy+boundary_padding_thickness)*pitch_c)+boundary_padding_thickness;
			
			for (int i=idy;i<m;i+=gridDim.y*blockDim.y){	
							
				for (int j=idx;j<k;j+=gridDim.x*blockDim.x){			
					F2 v=tex2D<F2>(A,i+boundary_padding_thickness+0.5,j+boundary_padding_thickness+0.5);
					F2 vlower=tex2D<F2>(A,i+boundary_padding_thickness+0.5,j-1+boundary_padding_thickness+0.5);
					F2 vupper=tex2D<F2>(A,i+boundary_padding_thickness+0.5,j+1+boundary_padding_thickness+0.5);
					F2 vleft=tex2D<F2>(A,i-1+boundary_padding_thickness+0.5,j+boundary_padding_thickness+0.5);
					F2 vright=tex2D<F2>(A,i+1+boundary_padding_thickness+0.5,j+boundary_padding_thickness+0.5);			
					
					C[j].x-=one_half_dx*(vleft.x-vright.x);
					C[j].y-=one_half_dy*(vupper.y-vlower.y);
				}		
				C=(F2*) ((char*)C+gridDim.y*blockDim.y*pitch_c);					
			}		
		
		}

		template<class F, class F2>
		cudaError_t subtract_gradient(F dy, F dx, int boundary_padding_thickness, int m, int k,F2* A_d, size_t pitch_a, F2* C_d, size_t pitch_c){
			//Create Resource descriptions
			cudaResourceDesc resDesc;
			memset(&resDesc,0,sizeof(resDesc));

			resDesc.resType = cudaResourceTypePitch2D;
			resDesc.res.pitch2D.devPtr=A_d;
			resDesc.res.pitch2D.width=k;
			resDesc.res.pitch2D.height=m;
			resDesc.res.pitch2D.pitchInBytes=pitch_a;
			resDesc.res.pitch2D.desc=cudaCreateChannelDesc<F2>(); 

			//Create Texture description
			cudaTextureDesc texDesc;
			memset(&texDesc,0,sizeof(texDesc));
			texDesc.normalizedCoords = false;
			texDesc.filterMode = cudaFilterModeLinear;
			texDesc.readMode=cudaReadModeElementType;
			texDesc.addressMode[0] = cudaAddressModeBorder;
			texDesc.addressMode[1] = cudaAddressModeBorder;

			cudaTextureObject_t A_tex;
			cudaError_t err=cudaCreateTextureObject(&A_tex, &resDesc, &texDesc, NULL);
			
			if (err !=cudaSuccess){
				return err;
			}				
			int threads_per_block_x=256;	
			int threads_per_block_y=4;	
			int blocks_x=ceil(static_cast<float>(k)/(threads_per_block_x));
			int blocks_y=ceil(static_cast<float>(m)/(threads_per_block_y));
			
			dim3 threads=dim3(threads_per_block_x,threads_per_block_y,1);
			dim3 blocks=dim3(blocks_x,blocks_y,1);
			k_subtract_gradient<F,F2><<<blocks,threads>>>(0.5*dy, 0.5*dx, boundary_padding_thickness, m, k,A_tex, C_d, pitch_c);		
			
			return cudaSuccess;
			
		}
		
		template
		cudaError_t subtract_gradient(float dy, float dx, int boundary_padding_thickness, int m, int k, float2* A_d, size_t pitch_a, float2* C_d, size_t pitch_c);
		
		__host__
		cudaError_t prolong_and_add(int m_p, int k_p, int m_r, int k_r, float2* dest, int pitch_dest, float2* src, int pitch_src);
	}
}
