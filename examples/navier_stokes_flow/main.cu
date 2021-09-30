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


void multigrid_example(float height, float width, int m, int k){

		//Problem parameters
		float viscousity=1;
		float dt=1;
		float dx=width/k;
		float dy=height/m;
		int multigrid_stages=1;
		int max_jacobi_iterations_per_stage[]={30,30,30,30,30,30,30,30,30,30};//maximum number of iterations per multigrid stage
		float jacobi_weight=1.0; //weight factor of the weighted Jacobi method
		float jacobi_tol=0.1; //if sum of squares of the residual goes below this value, the estimation of x in Ax=b terminates and returns DesalStatus::Success
		float sos_residual; // this saves the sum of squares of the residual
		
		

		//Allocate Vectors
		float2* U_old; //flow field vector
		float2* U_new; //flow field vector buffer
	
		size_t pitch_u_old;
		size_t pitch_u_new;
		
		cudaMallocPitch((void**)&U_new,&pitch_u_new,sizeof(float2)*k,m);
		cudaMallocPitch((void**)&U_old,&pitch_u_old,sizeof(float2)*k,m);	
		
		//Initialize grid
		cudaMemset(U_old,0.0,sizeof(float2)*k*m);	
		
		float2 u_val;
		
		u_val.x=1;
		u_val.y=2;

		//desal::cuda::fill_array_ascendingly2D_f32(m,k,1,U,pitch_u,0);
		desal::cuda::fill_array_uniformly2D<float2>(m,k,1,U_old,pitch_u_old,u_val);
		
		desal::cuda::navier_stokes_2D_device(dt, viscousity, 1, dy,dx,  m,k, U_old, pitch_u_old, U_new, pitch_u_new, max_jacobi_iterations_per_stage, multigrid_stages,jacobi_weight,  jacobi_tol, &std::cout);

		

	cudaFree(U_new);
	cudaFree(U_old);
}

int main(){
	size_t free_device_memory;
	size_t total_device_memory;
	cudaMemGetInfo(&free_device_memory,&total_device_memory);
	
	std::cout<<"This device has "<<total_device_memory/(1000000000.0)<<" Gigabytes of GPU memory\n";
	int x_points=20;
	int y_points=20;
	float height=1.0;
	float width=1.0;
	multigrid_example(height,width,y_points,x_points); //TODO: Test n=97
	std::cout<<"Berechnung beendet\n";
}