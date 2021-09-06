template<unsigned int THREADS_X_PER_BLOCK>
__device__
void d_warp_reduce_sum(volatile float* sdata, int ty, int tx){
	int index=ty*0.5*THREADS_X_PER_BLOCK+tx;
	if (THREADS_X_PER_BLOCK >=128){
		sdata[index]+=sdata[index+64];
		__syncthreads();
	}
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

template<unsigned int THREADS_X_PER_BLOCK>
__device__
void d_warp_reduce_sum(volatile float* sdata, int tx){
	sdata[tx]+=sdata[tx+64];
	sdata[tx]+=sdata[tx+16];
	sdata[tx]+=sdata[tx+8];
	sdata[tx]+=sdata[tx+4];
	sdata[tx]+=sdata[tx+2];	
	sdata[tx]+=sdata[tx+1];
}

//AX=B
template<unsigned int THREADS_X_PER_BLOCK,unsigned int THREADS_Y_PER_BLOCK>
__global__
void k_reduce_sos_fivepoint_stencil_float2(float alpha_inv, float beta, float boundary_offset, int m, int k, cudaTextureObject_t A,cudaTextureObject_t B, float* r, int stride_r){
	static __shared__ float sdata[static_cast<int>(0.5*THREADS_X_PER_BLOCK*THREADS_Y_PER_BLOCK)];

	int idy=blockIdx.y*blockDim.y+threadIdx.y;
	int idx=blockIdx.x*blockDim.x+threadIdx.x;
	int tx=threadIdx.x;
	int ty=threadIdx.y;
	
	float* r_ptr=r;
	
	
	for (int hs=0;hs<m;hs+=gridDim.y*blockDim.y){
		int fy=idy;
		int fx=idx;		
		r_ptr=r+hs*stride_r;
		for (int ws=0;ws<k;ws+=gridDim.x*blockDim.x){
			float2 v=tex2D<float2>(A,fx+boundary_offset,fy+boundary_offset);
			float2 vlower=tex2D<float2>(A,fx+boundary_offset,fy-1+boundary_offset);
			float2 vupper=tex2D<float2>(A,fx+boundary_offset,fy+1+boundary_offset);
			float2 vleft=tex2D<float2>(A,fx-1+boundary_offset,fy+boundary_offset);
			float2 vright=tex2D<float2>(A,fx+1+boundary_offset,fy+boundary_offset);
	
			float2 b=tex2D<float2>(B,fx+boundary_offset,fy+boundary_offset);
			float2 diff;
			diff.x=beta*(v.x-vleft.x-vright.x-vupper.x-vlower.x)*alpha_inv-b.x;
			diff.y=beta*(v.y-vleft.y-vright.y-vupper.y-vlower.y)*alpha_inv-b.y;
	
			sdata[ty*blockDim.x+tx]=diff.x*diff.x+diff.y*diff.y;
		
			v=tex2D<float2>(A,fx+blockDim.x+boundary_offset,fy+boundary_offset);
			vlower=tex2D<float2>(A,fx+blockDim.x+boundary_offset,fy-1+boundary_offset);
			vupper=tex2D<float2>(A,fx+blockDim.x+boundary_offset,fy+1+boundary_offset);
			vleft=tex2D<float2>(A,fx+blockDim.x-1+boundary_offset,fy+boundary_offset);
			vright=tex2D<float2>(A,fx+blockDim.x+1+boundary_offset,fy+boundary_offset);
			
			b=tex2D<float2>(B,fx+blockDim.x+boundary_offset,fy+boundary_offset);
			
			diff.x=beta*(v.x-vleft.x-vright.x-vupper.x-vlower.x)*alpha_inv-b.x;
			diff.y=beta*(v.y-vleft.y-vright.y-vupper.y-vlower.y)*alpha_inv-b.y;
	
			sdata[ty*0.5*blockDim.x+tx]+=diff.x*diff.x+diff.y*diff.y; //TODO: Add second element
			
			__syncthreads();
			
			if (THREADS_X_PER_BLOCK>=512){
				if (tx<256){
				sdata[ty*0.5*blockDim.x+tx]+=sdata[ty*0.5*blockDim.x+tx+256];							
				}
				__syncthreads();
			}
			if (THREADS_X_PER_BLOCK>=256){
				if (tx<128){
					sdata[ty*0.5*blockDim.x+tx]+=sdata[ty*0.5*blockDim.x+tx+128];				
				}
				__syncthreads();
			}	
			if (THREADS_X_PER_BLOCK>=128){
				if (tx<64){
					sdata[ty*0.5*blockDim.x+tx]+=sdata[ty*0.5*blockDim.x+tx+64];				
				}
				__syncthreads();
			}
			if (tx<32){
				d_warp_reduce_sum<THREADS_X_PER_BLOCK>(sdata,ty,tx);
				
				//TODO: Evaluate possibility of bank conflicts
				if (tx==0){
					r_ptr[ty]=sdata[static_cast<int>(ty*0.5*THREADS_X_PER_BLOCK)];
				}
			}
			__syncthreads();			
			fx+=ws;
			r_ptr+=blockDim.y;
		}
		fy+=hs;
		r_ptr=r+gridDim.y*blockDim.y;
	}

}

template<unsigned int THREADS_X_PER_BLOCK, class F>
__global__
void k_reduce_sum(int n, float* r, int stride_r){
	static __shared__ float sdata[static_cast<int>(0.5*THREADS_X_PER_BLOCK)];

	int idx=blockIdx.x*blockDim.x+threadIdx.x;
	int tx=threadIdx.x;

	for (int hs=0;hs<n;hs+=gridDim.x*blockDim.x){
			int index=blockIdx.x*2*blockDim.x+tx;
			sdata[tx]=r[index]+r[index+blockDim.x];
			
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
				
				if (tx==0){
					r[tx]=sdata[0];
				}
			}
			__syncthreads();			
			r+=hs;
	}
}