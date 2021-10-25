#include <stdio.h>
#include "../../../../src/gpu/cuda/solvers/poisson_multigrid.h"
#include "utility.h"

namespace desal{
	
	namespace cuda{

		__global__
		void print_matrix_k(int m,int k, float* M, int stride_col,int stride_row,char name){
			printf("%c:\n",name);
			for (int i=0;i<m;i++){
				for (int j=0;j<k;j++){
					printf("%.1f ",M[i*stride_col+j*stride_row]);
				}
				printf("\n");
			}	
		}

		__global__
		void print_vector_field_k(int m,int k, float2* M, int pitch,char name){
			printf("%c:\n",name);
			for (int i=0;i<m;i++){
				float2* current_row=(float2*)((char*)M + i*pitch);
				for (int j=0;j<k;j++){
					printf("(%.1f,%.1f) ",current_row[j].x,current_row[j].y);
				}
				printf("\n");
			}	
		}

		bool test_mg_vc_poisson_2D_f32_zero_B(char* error_message=nullptr){
			
			for (int j=4;j<8;j++){
			
			int m=1<<(3+j);
			int k=1<<(3+j);
			float2* U; //flow field vector
			float2* B; //flow field vector
			float sos_residual_prev=1e30;
				for (int i=1;i<=(j+2);i++){	
					float height=1;
					float width=1;
					
					//Problem parameters
					float dt=1;
					float dx=width/k;
					float dy=height/m;
					
					float v=1.0; //viscousity coefficient
					float alpha=(dx*dy)/(v*dt); //see manual for details
					float gamma=alpha; //see manual for details
					float eta=4.0; //see manual for details
					
					//Allocate Device Memory
					size_t pitch_u;
					size_t pitch_b;
						
					cudaMallocPitch((void**)&U,&pitch_u,sizeof(float2)*k,m);
					cudaMallocPitch((void**)&B,&pitch_b,sizeof(float2)*k,m);

					cudaMemset2D(B,pitch_b,0.0,k*sizeof(float2),m);	
					cudaMemset2D(U,pitch_u,0.0,k*sizeof(float2),m);	
					
					float2 u_val;
					u_val.x=-2;
					u_val.y=1;
					
					fill_array_uniformly2D<float2>(m,k,1,U,pitch_u,u_val);
					
					int multigrid_stages=i;
					int max_jacobi_iterations_per_stage[]={30,30,30,30,30,30,30,30,30,30};//maximum number of iterations per multigrid stage
					float jacobi_weight=1.0; //weight factor of the weighted Jacobi method
					float tol=0.1; //if sum of squares of the residual goes below this value, the estimation of x in Ax=b terminates and returns DesalStatus::Success
					float sos_residual; // this saves the sum of squares of the residual
				
					auto res=desal::cuda::mg_vc_poisson_2D_device<float, float2,std::ostream>(alpha, gamma,eta, 1, m,k, B, pitch_b, U, pitch_u,max_jacobi_iterations_per_stage,multigrid_stages, jacobi_weight, tol, &sos_residual,&std::cout);
					cudaFree(B);
					cudaFree(U);					
					printf("sos_residual %f in iter %d\n",sos_residual, i);
					if (sos_residual<sos_residual_prev){
						sos_residual_prev=sos_residual;
					}
					else{
						return false;
					}
					
					if (sos_residual<0){
						return false;
					}
			
				}		
			
			}
	
			return true;	
		}
	}
}