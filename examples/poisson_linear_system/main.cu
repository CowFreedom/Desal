/** \file main.cpp
 * This example calculates (I-M)u=b, with I being an identity and M a forward difference matrix. In addition, u,b are vectors specifying the grid points.
 * The calculation is conducted via a V-Cycle multigrid algorith with multiple stages.
 * It shall be noted, that if the number of stages is increased, the resulting residual error decreases.
 * For minimum error, the number of maximum iterations of each stage should be chosen until that the squared residual barely changes within Jacobi iterations.
 */
 
#include <iostream>
#include <iomanip>
#include <chrono>
#include <fstream>
#include <string>
#include "../../src/gpu/cuda/error_handling.h"
#include "../../src/gpu/cuda/solvers/poisson_multigrid.h"
#include "../../diagnostics/correctness/gpu/cuda/utility.h"


__global__
void k_set_inner_points(float2 val, int m, int k, float2* U, size_t pitch_u){
	m-=1;
	k-=1;
	int idx=blockIdx.x*blockDim.x+threadIdx.x+1;			
	int idy=blockIdx.y*blockDim.y+threadIdx.y+1;
	U=(float2*)((char*)U+idy*pitch_u);

	for (int i=idy;i<m;i+=blockDim.y*gridDim.y){
		for (int j=idx; j<k;j+=blockDim.x*gridDim.x){
			U[j]=val;
		}
		U=(float2*)((char*)U+pitch_u);		
	}
}


void multigrid_example(float height, float width, int m, int k){
	
	float dt=0.0125;
	float dx=width/k;
	float dy=height/m;
	int multigrid_stages=5;
	int max_jacobi_iterations_per_stage[]={30,30,30,30,30,30,10000,10000,2000,2000};//maximum number of iterations per multigrid stage
	float jacobi_weight=1; //weight factor of the weighted Jacobi method
	float tol=2*(1e-1)*m*k; //if sum of squares of the residual goes below this value, the estimation of x in Ax=b terminates and returns DesalStatus::Success
	float early_termination_tol=1; //if residual_prev/residual_current is above early_termination_tol, then a Jacobi Iteration finishes early due to diminishing returns. If set to 1 then the iterations will not terminate early
	float sos_residual=-1; // this saves the sum of squares of the residual

	float v=0.01; //conductivity coefficient
	float alpha=(dx*dy)/(v*dt); //see manual for details
	float gamma=alpha; //see manual for details
	float eta=4.0; //see manual for details
	
	//Allocate Vectors
	float2* B; //flow field vector
	float2* U; //flow field vector buffer	

	size_t pitch_b;
	size_t pitch_u;
	
	cudaMallocPitch((void**)&U,&pitch_u,sizeof(float2)*k,m);
	cudaMallocPitch((void**)&B,&pitch_b,sizeof(float2)*k,m);	
	
	//Initialize grid
	float2 u_val;
	u_val.x=100;
	u_val.y=-100;	
	cudaMemset2D(B,pitch_b,0.0,sizeof(float2)*k,m);	

	int iter=5;
	
	int timevals=0;
	
	for (int i=0; i<iter;i++){	
		cudaMemset2D(U,pitch_u,0.0,sizeof(float2)*k,m);		
		k_set_inner_points<<<dim3(ceil(m/32.0),1,1),dim3(32,1,1)>>>(u_val,m,k,U,pitch_u);
	//	desal::cuda::print_vector_field_k2<<<1,1>>>(m,k, B, pitch_b,'B');
	//	desal::cuda::print_vector_field_k2<<<1,1>>>(m,k, U, pitch_u,'U');
		
		//The time taken to solve the linear system of equations is printed
		std::cout<<std::setprecision(9)<<std::fixed;
		
		// Get starting timepoint 
		auto start = std::chrono::high_resolution_clock::now(); 
		//Function to measure

		desal::cuda::DesalStatus res=desal::cuda::mg_vc_poisson<float, float2,std::ostream>(alpha, gamma,eta, 1, m,k, B, pitch_b, U, pitch_u,max_jacobi_iterations_per_stage,multigrid_stages, jacobi_weight, tol,early_termination_tol, &sos_residual);		
		// Get ending timepoint 
		auto stop = std::chrono::high_resolution_clock::now(); 
		
		auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start); 
		timevals+=duration.count();
		std::cout<<std::right<<"Time [ms]"<<std::setw(40)<<"V-Cycle Stages"<<std::setw(30)<<"Sum of Squares of Residual\n";

		std::cout<<std::right<<duration.count()<<std::setw(40)<<multigrid_stages<<std::setw(30)<<sos_residual<<"\n";
		if (res!=desal::cuda::DesalStatus::Success){
			if (res==desal::cuda::DesalStatus::CUDAError){
				std::cout<<"CudaError\n";
				return;
			}
			else if (res==desal::cuda::DesalStatus::InvalidParameters){
				std::cout<<"Invalid parameters supplied to the multigrid solver\n";
				return;
			}
			else{
				std::cout<<"Error could not successfully be reduced\n";
			}
			
		}		
		
	}
	std::cout<<"Avg. duration:" <<timevals/5<<"\n";
	cudaFree(B);
	cudaFree(U);	
}

int main(){
	size_t free_device_memory;
	size_t total_device_memory;
	cudaMemGetInfo(&free_device_memory,&total_device_memory);
	
	std::cout<<"This device has "<<total_device_memory/(1000000000.0)<<" Gigabytes of GPU memory\n";
	float height=10.0;
	float width=10.0;
	int m=1000;
	int k=1000;
	multigrid_example(height,width,m,k);



}