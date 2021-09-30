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
#include "../../src/gpu/cuda/solvers/poisson_multigrid.h"
#include "../../diagnostics/correctness/gpu/cuda/utility.h"


void multigrid_example(int m, int k, int reps){


	for (int i=1;i<=reps;i++){	
		float2* U; //flow field vector
		float2* B; //flow field vector

		int height=m;
		int width=k;
		
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

		cudaMemcpy2D(B,pitch_b,U,pitch_u,k*sizeof(float2),m,cudaMemcpyDeviceToDevice);
		cudaMemset2D(B,pitch_b,0.0,(k-2)*sizeof(float2),m-2);	
		
		//Create b and starting vector x0 in Ax0=b
		//float2 u_val;
		float2 b_val;
		b_val.x=0;
		b_val.y=0;
		//u_val.x=1;
		//u_val.y=1;
		//fill_array_uniformly2D<float2>(m,k,1,U,pitch_u,u_val);
		desal::cuda::fill_array_ascendingly2D_f32(m,k,1,U,pitch_u,0);
		desal::cuda::fill_array_uniformly2D<float2>(m,k,1,B,pitch_b,b_val);
		
		
		int multigrid_stages=i;
		int max_jacobi_iterations_per_stage[]={30,30,30,30,30,30,30,30,30,30};//maximum number of iterations per multigrid stage
		float jacobi_weight=1.0; //weight factor of the weighted Jacobi method
		float tol=0.1; //if sum of squares of the residual goes below this value, the estimation of x in Ax=b terminates and returns DesalStatus::Success
		float sos_residual; // this saves the sum of squares of the residual
		
		//The time taken to solve the linear system of equations is printed
		// Get starting timepoint 
		auto start = std::chrono::high_resolution_clock::now(); 
		//Function to measure

		std::cout<<std::setprecision(2)<<std::fixed;
		auto res=desal::cuda::mg_vc_poisson_2D_device<float, float2,std::ostream>(alpha, gamma,eta, 1, m,k, B, pitch_b, U, pitch_u,max_jacobi_iterations_per_stage,multigrid_stages, jacobi_weight, tol, &sos_residual,&std::cout);
		// Get ending timepoint 
		auto stop = std::chrono::high_resolution_clock::now(); 
		 
		auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start); 
		std::cout<<"\n"<<std::right<<"Problemsize"<<std::right<<std::setw(20)<<"Time [ms]"<<std::setw(40)<<"V-Cycle Stages"<<std::setw(30)<<"Sum of Squares of Residual\n";

		std::cout<<std::right<<m<<std::setw(20)<<std::right<<duration.count()<<std::setw(40)<<multigrid_stages<<std::setw(30)<<sos_residual<<"\n";
		
		if (sos_residual<tol && sos_residual>=0){
			std::cout<<"The sum of squares error could be successfully reduced below the set tolerance\n\n";
		}	
		else if (res==desal::cuda::DesalStatus::InvalidParameters){
			std::cout<<"Invalid parameters\n";
			return;
		}
		else{
			std::cout<<"The sum of squares error could not be successfully reduced below the set tolerance. Try more multigrid stages, more iterations per stage or a higher error tolerance\n\n";
		}

		cudaFree(B);
		cudaFree(U);
	
			
	}
}

int main(){
	size_t free_device_memory;
	size_t total_device_memory;
	cudaMemGetInfo(&free_device_memory,&total_device_memory);
	
	std::cout<<"This device has "<<total_device_memory/(1000000000.0)<<" Gigabytes of GPU memory\n";
	int m=40;
	int k=10;
	multigrid_example(m,k, 1);



}