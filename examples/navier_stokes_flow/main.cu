/** \file main.cpp
 * This example calculates (I-M)u=b, with I being an identity and M a forward difference matrix. In addition, u,b are vectors specifying the grid points.
 * The calculation is conducted via a V-Cycle multigrid algorith with multiple stages.
 * It shall be noted, that if the number of stages is increased, the resulting residual error decreases.
 * For maximum effect, the number of maximum iterations of each stage should be chosen such that the squared residual barely changes within Jacobi iterations.
 */
 
#include <iostream>
#include <iomanip>
#include <chrono>
#include "../../src/gpu/cuda/error_handling.h"
#include "../../src/gpu/cuda/solvers/navier_stokes.h"
#include "../../diagnostics/correctness/gpu/cuda/utility.h"

void navier_stokes_example(float height, float width, int m, int k){

		//Problem parameters
		float viscousity=1;
		float dt=1;
		float dx=width/k;
		float dy=height/m;
		int multigrid_stages=1;
		int max_jacobi_iterations_per_stage[]={30,30,30,30,30,30,30,30,30,30};//maximum number of iterations per multigrid stage
		float jacobi_weight=1.0; //weight factor of the weighted Jacobi method
		float jacobi_tol=0.1; //if sum of squares of the residual goes below this value, the estimation of x in Ax=b terminates and returns DesalStatus::Success
		

		//Allocate Vectors
		float2* U_old; //flow field vector
		float2* U_new; //flow field vector buffer
		float2* F; //flow field vector buffer
		
	
		size_t pitch_u_old;
		size_t pitch_u_new;
		size_t pitch_f;
		
		cudaMallocPitch((void**)&U_new,&pitch_u_new,sizeof(float2)*k,m);
		cudaMallocPitch((void**)&U_old,&pitch_u_old,sizeof(float2)*k,m);	
		cudaMallocPitch((void**)&F,&pitch_f,sizeof(float2)*k,m);	
		
		//Initialize grid
		cudaMemset2D(U_old,pitch_u_old,0.0,sizeof(float2)*k,m);	
		
		float2 u_val;	
		u_val.x=1;
		u_val.y=-1;
		
		float2 f_val;
		f_val.x=0;
		f_val.y=0;

		//desal::cuda::fill_array_ascendingly2D_f32(m,k,1,U,pitch_u,0);
		//desal::cuda::fill_array_uniformly2D<float2>(m,k,1,U_old,pitch_u_old,u_val);
		
			
		desal::cuda::fill_array_uniformly2D<float2>(m,k,1,U_old,pitch_u_old,u_val);
		desal::cuda::fill_array_uniformly2D<float2>(m,k,1,U_old,pitch_u_old,u_val);
		desal::cuda::print_vector_field_k2<<<1,1>>>(m,k, U_old, pitch_u_old,'O');
		desal::cuda::navier_stokes_2D_device<float,float2,std::ostream>(dt, viscousity, 1, dy,dx,  m,k, U_old,pitch_u_old, F, pitch_f, U_new, pitch_u_new, max_jacobi_iterations_per_stage, multigrid_stages,jacobi_weight,  jacobi_tol, &std::cout);
		desal::cuda::print_vector_field_k2<<<1,1>>>(m,k, U_new, pitch_u_new,'N');
		

	cudaFree(U_new);
	cudaFree(U_old);
}

int main(){
	size_t free_device_memory;
	size_t total_device_memory;
	cudaMemGetInfo(&free_device_memory,&total_device_memory);
	
	std::cout<<"This device has "<<total_device_memory/(1000000000.0)<<" Gigabytes of GPU memory\n";
	int x_points=21;
	int y_points=21;
	float height=10.0;
	float width=10.0;
	navier_stokes_example(height,width,y_points,x_points); //TODO: Test n=97
	std::cout<<"Berechnung beendet\n";
}