#include <stdio.h>
#include "../../src/gpu/cuda/solvers/poisson_multigrid.h"
#include "../../diagnostics/correctness/gpu/cuda/utility.h"

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

bool test(int n, int reps, char* error_message=nullptr){
	for (int i=1;i<=reps;i++){	
		n=i*n;
		float2* U; //flow field vector
		float2* B; //flow field vector
		float* r; // stores diffused velocity field
		int width=n;
		int height=n;
		int k=n;
		int m=n;
		
		float dt=1;
		float dx=width/k;
		float dy=height/m;
		
		float v=1.0; //Viscousity coefficient
		float alpha=(dx*dy)/(v*dt);
		//float beta=4.0+alpha;	
		float gamma=alpha;
		float eta=4.0;
		size_t pitch_u;
		size_t pitch_b;
		
		//Allocate Device Memory	
		cudaMallocPitch((void**)&U,&pitch_u,sizeof(float2)*k,m);
		cudaMallocPitch((void**)&B,&pitch_b,sizeof(float2)*k,m);

		cudaMemcpy2D(B,pitch_b,U,pitch_u,k*sizeof(float2),m,cudaMemcpyDeviceToDevice);
		cudaMemset2D(B,pitch_b,0.0,(k-2)*sizeof(float2),m-2);	

		cudaMalloc((void**)&r,sizeof(float)*k*m);
		cudaMemset(r,0.0,sizeof(float)*k*m);
		
		float2 u_val;
		float2 b_val;
		b_val.x=0;
		b_val.y=0;
		u_val.x=1;
		u_val.y=1;
		//fill_array_uniformly2D<float2>(m,k,1,U,pitch_u,u_val);
		fill_array_ascendingly2D_f32(m,k,1,U,pitch_u,0);
		fill_array_uniformly2D<float2>(m,k,1,B,pitch_b,b_val);
		
		float exact_result=0;
		print_vector_field_k<<<1,1>>>(m,k,U, pitch_u,'U');
		mg_vc_poisson_2D_f32_device(alpha, gamma,eta, 1, n, B, pitch_b, U, pitch_u);

			
	//	reduce_sum_of_squares_poisson_field_residual_f32_device(alpha,beta,1, n,U,pitch_u, B, pitch_b, r);
	//	
		//print_vector_field_k<<<1,1>>>(m,k,B, pitch_b,'B');
	//	print_matrix_k<<<1,1>>>(1,m*k, r, m,1,'r');		
	//	printf("Result is:%f\n",exact_result);
		
		float gpu_result;
	//	cudaMemcpy(&gpu_result,r,sizeof(float)*1,cudaMemcpyDeviceToHost);
		
		cudaFree(r);
		cudaFree(U);
	
			
	}
	return true;	
}

int main(){

	test(5, 1);

}