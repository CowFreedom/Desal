//#include "utility.h"
#include <stdio.h>

namespace desal{
	namespace cuda{

		template<class F>
		__global__ 
		void k_fill_array_ascendingly(int n, F* arr, int stride,int offset=0){
			int index=blockIdx.y*blockDim.x+blockIdx.x*blockDim.x+threadIdx.x;
			
			for (int i=0;i<n;i+=gridDim.x*blockDim.x){
				index+=i;
				arr[index]=index+1+offset;
			}
		}

		template<class F>
		__global__ 
		void k_fill_array_descendingly(int n, F* arr, int stride, int offset=0){
			int index=blockIdx.y*blockDim.x+blockIdx.x*blockDim.x+threadIdx.x;
			
			for (int i=0;i<n;i+=gridDim.x*blockDim.x){
				index+=i;
				arr[index]=n-index+offset;
			}
		}

		template<class F>
		__host__
		void fill_array_ascendingly(int n, F* arr, int stride,int offset=0){
			F threads=32;
			int blocks=ceil(n/threads);
			k_fill_array_ascendingly<F><<<blocks,threads>>>(n,arr,stride,offset);
		}

		template<class F>
		__host__
		void fill_array_descendingly(int n, F* arr, int stride,int offset=0){
			F threads=32;
			int blocks=ceil(n/threads);
			k_fill_array_descendingly<F><<<blocks,threads>>>(n,arr,stride,offset);
		}

		__host__
		void fill_array_ascendingly_f32(int n, float* arr, int stride,int offset=0){
			fill_array_ascendingly<float>(n,arr,stride,offset);
		}

		__host__
		void fill_array_descendingly_f32(int n, float* arr, int stride,int offset=0){
			fill_array_descendingly<float>(n,arr,stride,offset);
		}

		__host__
		void fill_array_ascendingly_f64(int n, double* arr, int stride,int offset=0){
			double threads=32;
			int blocks=ceil(n/threads);
			fill_array_descendingly<double>(n,arr,stride,offset);
		}

		__host__
		void fill_array_descendingly_f64(int n, double* arr, int stride,int offset=0){
			double threads=32;
			int blocks=ceil(n/threads);
			fill_array_descendingly<double>(n,arr,stride,offset);
		}


		template<class F2>
		__global__
		void k_fill_array_ascendingly2D_field(int m,int k,int boundary_padding_thickness, F2* U,int pitch_u, int offset=1){
			m-=2*boundary_padding_thickness;
			k-=2*boundary_padding_thickness;
			U=(F2*)((char*)U+boundary_padding_thickness*pitch_u);;
			for (int i=blockIdx.y*blockDim.y+threadIdx.y;i<m;i+=gridDim.y*blockDim.y){
				F2* temp=(F2*)((char*)U+i*pitch_u);
				for (int j=blockIdx.x*blockDim.x+threadIdx.x+boundary_padding_thickness;j<k;j+=gridDim.y*blockDim.x){
					F2 val;
					val.x=i*k+j+offset;
					val.y=i*k+j+offset;
					temp[j+boundary_padding_thickness]=val;
				}
			}
		}
		

		__host__
		void fill_array_ascendingly2D_f32(int m, int k, int boundary_padding_thickness, float2* U, int pitch_u, int offset=0){
			float threads_x=32;
			float threads_y=32;
			dim3 threads=dim3(threads_x,threads_y,1);
			dim3 blocks=dim3(ceil(k/threads_x),ceil(m/threads_y),1);

			k_fill_array_ascendingly2D_field<float2><<<blocks,threads>>>(m,k,boundary_padding_thickness,U,pitch_u,offset);	
		}


		__global__
		void k_check_boundary(int n,float2* A, int pitch,float val){
			float2* down=(float2*) ((char*)A+(n-1)*pitch);


			for (int i=0;i<n;i++){
					
					float2* vert=(float2*) ((char*)A+i*pitch);
			
					if (A[i].x!=val ||down[i].x!=val){
						printf("Horizontal boundary x error with values %f and %f\n", A[i].x, down[i].x);
				
					}
					
					if (vert[0].x!=val || vert[(n-1)].x!=val){
						printf("vertical boundary x error with values %f and %f\n", vert[0].x, vert[(n-1)].x);
					}
					
					
					if (A[i].y=val || down[i].y!=val){
						printf("Horizontal boundary y error\n");
					}
					
					if (vert[0].y!=val || vert[(n-1)].y!=val){
						printf("vertical boundary y error\n");
					}
					
			}
			
		}
	}
}