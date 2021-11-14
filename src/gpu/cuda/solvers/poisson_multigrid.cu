#include <stdio.h>
#include "poisson_multigrid.h"
#include "../error_handling.h"
#include "../reductions.h"
#include "../transformations.h"

namespace desal{

	namespace cuda{
	
		//m: height of interior points k: width of interior plus boundary points
		__global__
		void k_jacobi_poisson(float weight, float alpha, float beta_inv, int boundary_padding_thickness, int m, int k, cudaTextureObject_t X_old, float2* X_new, int pitch_x, cudaTextureObject_t B){
			m-=boundary_padding_thickness;
			k-=boundary_padding_thickness;
			int idy=blockIdx.y*blockDim.y+threadIdx.y+boundary_padding_thickness;
			int idx=blockIdx.x*blockDim.x+threadIdx.x+boundary_padding_thickness;
			
			X_new=(float2*) ((char*)X_new+idy*pitch_x);	
			
			for(int i=idy;i<m;i+=gridDim.y*blockDim.y){
				for(int j = idx; j<k; j+=gridDim.x*blockDim.x){
					float2 x=tex2D<float2>(X_old,j+0.5,i+0.5);
					float2 xupper=tex2D<float2>(X_old,j+0.5,i+1+0.5);
					float2 xlower=tex2D<float2>(X_old,j+0.5,i-1+0.5);
					float2 xright=tex2D<float2>(X_old,j+1+0.5,i+0.5);
					float2 xleft=tex2D<float2>(X_old,j-1+0.5,i+0.5);						
					float2 b=tex2D<float2>(B,j+0.5,i+0.5);

					X_new[j].x=(1.0-weight)*x.x+weight*beta_inv*(xlower.x+xupper.x+xleft.x+xright.x+alpha*b.x);	
					X_new[j].y=(1.0-weight)*x.y+weight*beta_inv*(xlower.y+xupper.y+xleft.y+xright.y+alpha*b.y);									
				}
				X_new=(float2*) ((char*)X_new+gridDim.y*blockDim.y*pitch_x);
			}
		}
		
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