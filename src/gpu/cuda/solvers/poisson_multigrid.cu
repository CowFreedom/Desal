#include <stdio.h>
#include "poisson_multigrid.h"
#include "../error_handling.h"
#include "../reductions.h"
#include "../transformations.h"

namespace desal{

	namespace cuda{
	
		//m: height of interior points k: width of interior plus boundary points
		template<class F, class F2>
		__global__
		void k_jacobi_poisson_2D(F weight, F alpha, F beta_inv, int boundary_padding_thickness, int m, int k, cudaTextureObject_t X_old, F2* X_new, int pitch_x, cudaTextureObject_t B){
			m-=2*boundary_padding_thickness;
			k-=2*boundary_padding_thickness;
			int idy=blockIdx.y*blockDim.y+threadIdx.y;
			int idx=blockIdx.x*blockDim.x+threadIdx.x;
			
			float2* X_ptr=X_ptr=(float2*) ((char*)X_new+(idy+boundary_padding_thickness)*pitch_x);	

			for(int i=idy;i<m;i+=gridDim.y*blockDim.y){
					
				for(int j = idx; j<k; j+=gridDim.x*blockDim.x){
					float2 x=tex2D<float2>(X_old,j+boundary_padding_thickness+0.5,i+boundary_padding_thickness+0.5);
					float2 xupper=tex2D<float2>(X_old,j+boundary_padding_thickness+0.5,i+1+boundary_padding_thickness+0.5);
					float2 xlower=tex2D<float2>(X_old,j+boundary_padding_thickness+0.5,i-1+boundary_padding_thickness+0.5);
					float2 xright=tex2D<float2>(X_old,j+1+boundary_padding_thickness+0.5,i+boundary_padding_thickness+0.5);
					float2 xleft=tex2D<float2>(X_old,j-1+boundary_padding_thickness+0.5,i+boundary_padding_thickness+0.5);						
					float2 b=tex2D<float2>(B,j+boundary_padding_thickness+0.5,i+boundary_padding_thickness+0.5);
				//	printf("Val:(%f,%f)index: %d,%d\n",x.x,x.y, i,j);
					X_ptr[j+boundary_padding_thickness].x=(1.0-weight)*x.x+weight*beta_inv*(xlower.x+xupper.x+xleft.x+xright.x+alpha*b.x);	
					X_ptr[j+boundary_padding_thickness].y=(1.0-weight)*x.y+weight*beta_inv*(xlower.y+xupper.y+xleft.y+xright.y+alpha*b.y);									
				}
				X_ptr=(float2*) ((char*)X_ptr+(gridDim.y*blockDim.y)*pitch_x);	 //check if i+1 is correct	
			}
		}
		
		template
		__global__
		void k_jacobi_poisson_2D(float weight, float alpha, float beta_inv, int boundary_padding_thickness, int m, int k, cudaTextureObject_t X_old, float2* X_new, int pitch_x, cudaTextureObject_t B);

		__global__
		void print_vector_field_k2(int m,int k, float2* M, int pitch,char name, int iteration){
			if (iteration!=0){
				printf("iteration: %d\n",iteration);
			}
			printf("%c:\n",name);
			for (int i=0;i<m;i++){
				float2* current_row=(float2*)((char*)M + i*pitch);
				for (int j=0;j<k;j++){
					printf("(%.1f,%.1f) ",current_row[j].x,current_row[j].y);
				}
				printf("\n");
			}	
		}

		__global__
		void print_vector_k2(int m, float* M,char name, int iteration=0){
			printf("%c:\n",name);
			if (iteration!=0){
				printf("iteration: %d\n",iteration);
			}
			
				for (int j=0;j<m;j++){
					printf("%.1f ",M[j]);
				}
				printf("\n");
		}
		
		
	}
}