/** \file main.cpp
 * This example calculates (I-M)u=b, with I being an identity and M a forward difference matrix. In addition, u,b are vectors specifying the grid points.
 * The calculation is conducted via a V-Cycle multigrid algorith with multiple stages.
 * It shall be noted, that if the number of stages is increased, the resulting residual error decreases.
 * For maximum effect, the number of maximum iterations of each stage should be chosen such that the squared residual barely changes within Jacobi iterations.
 */
 
#include <iostream>
#include <iomanip>
#include <chrono>
#include <fstream>
#include <string>
#include "../../src/gpu/cuda/error_handling.h"
#include "../../src/gpu/cuda/solvers/poisson_multigrid.h"
#include "../../diagnostics/correctness/gpu/cuda/utility.h"

void write_data_to_file(int m, int k, float2* U_d, size_t pitch, std::string path, std::string name, int iter){
	std::ofstream file;
	file.open(path+"/"+name+std::to_string(iter)+".csv");
	float2* temp;
	temp=new float2[m*k];
	
	cudaMemcpy2D(temp,sizeof(float2)*k,U_d,pitch,k*sizeof(float2),m,cudaMemcpyDeviceToHost);
	
	for (int i=0;i<m;i++){
		for (int j=0;j<k-1;j++){
			file<<std::to_string(temp[i*k+j].x)<<",";
			}
			file<<std::to_string(temp[i*k+k-1].x)<<"\n";
	}
	delete[] temp;
	file.close();
}

__device__
float a(float t){
	return 20;
}


__global__
void k_set_inner_points(float boundary_temperature, float interior_temperature, int m, int k, float height, float width, float2* U, size_t pitch_u){
	m-=1;
	k-=1;
	int idx=blockIdx.x*blockDim.x+threadIdx.x+1;			
	int idy=blockIdx.y*blockDim.y+threadIdx.y+1;
	
	U=(float2*)((char*)U+idy*pitch_u);

	float2 bu;
	bu.x=0.5*k+0.5*(1/width)*k;
	bu.y=0.5*m+0.5*(1/height)*m;
	float2 bl;
	bl.x=0.5*k-0.5*(1/width)*k;
	bl.y=0.5*m-0.5*(1/height)*m;

	for (int i=idy;i<m;i+=blockDim.y*gridDim.y){
		for (int j=idx; j<k;j+=blockDim.x*gridDim.x){
			
			float dx=max(bl.x-j,j-bu.x);
			float dy=max(bl.y-i,i-bu.y);
			dx=max(0.0,dx);
			dy=max(0.0,dy);
		
			U[j].x=boundary_temperature+(2*(interior_temperature-boundary_temperature)*0.5*(1+tanh(-(dx*dx+dy*dy))));
			U[j].y=0;
		}
		U=(float2*)((char*)U+pitch_u);	
	}
}

template<class F2>
__global__
void k_set_boundary_val(float t, int m, int k, F2* U, size_t pitch_u){
	int idx=blockIdx.x*blockDim.x+threadIdx.x;			

	F2* U_neighbor=(F2*) ((char*)U+pitch_u);	
	for (int i=idx;i<k;i+=gridDim.x*blockDim.x){
			U[i].x=a(t);
	}	
	float2* U_ptr=(F2*) ((char*)U+(idx)*pitch_u);	
	for (int i=idx;i<m;i+=gridDim.x*blockDim.x){	
		U_ptr[0].x=a(t);
		U_ptr[k-1].x=a(t);
		U_ptr=(F2*) ((char*)U_ptr+pitch_u);	
	}	
	U_neighbor=(F2*) ((char*)U+(m-2)*pitch_u);
	U_ptr=(F2*) ((char*)U_neighbor+pitch_u);	
	
	for (int i=idx;i<k;i+=gridDim.x*blockDim.x){
		U_ptr[i].x=a(t);	
	}
}

void multigrid_example(float interior_temperature, float height, float width, int m, int k, float tend){

	float dt=0.0125;
	float dx=width/k;
	float dy=height/m;
	int multigrid_stages=5;
	int max_jacobi_iterations_per_stage[]={5000,5000,4000,5000,10000,10000,10000,10000,2000,2000};//maximum number of iterations per multigrid stage
	float jacobi_weight=1; //weight factor of the weighted Jacobi method
	float tol=1e-3*((m-1)*(k-1)); //if sum of squares of the residual goes below this value, the estimation of x in Ax=b terminates and returns DesalStatus::Success
	float early_termination_tol=1; //if residual_prev/residual_current is above early_termination_tol, then a Jacobi Iteration finishes early due to diminishing returns. If set to 1 then the iterations will not terminate early
	float sos_residual=-1; // this saves the sum of squares of the residual

	float v=0.525; //conductivity coefficient
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
	
	cudaMemset2D(B,pitch_b,0.0,sizeof(float2)*k,m);	
	
	cudaMemset2D(U,pitch_u,0.0,sizeof(float2)*k,m);	
	float boundary_temperature=20;
	k_set_boundary_val<float2><<<dim3(ceil(m/32.0),1,1),dim3(32,1,1)>>>(20, m, k, U, pitch_u);
	k_set_boundary_val<float2><<<dim3(ceil(m/32.0),1,1),dim3(32,1,1)>>>(20, m, k, B, pitch_b);
	//k_set_inner_points<<<dim3(ceil(m/32.0),1,1),dim3(32,1,1)>>>(m,k,U,pitch_u);
	k_set_inner_points<<<dim3(ceil(m/32.0),1,1),dim3(32,1,1)>>>(boundary_temperature, interior_temperature,m,k,height, width, B,pitch_b);


//desal::cuda::print_vector_field_k2<<<1,1>>>(m,k, B, pitch_b,'B');
	//desal::cuda::print_vector_field_k2<<<1,1>>>(m,k, U, pitch_u,'U');
	write_data_to_file(m, k, B, pitch_b, "output/", "output", 0);	
	int iter=1;

	for (float t=dt;t<=tend;t+=dt){	
		//The time taken to solve the linear system of equations is printed
		std::cout<<std::setprecision(3)<<std::fixed;
		k_set_boundary_val<float2><<<dim3(ceil(m/32.0),1,1),dim3(32,1,1)>>>(t, m, k, B, pitch_b);	
		k_set_boundary_val<float2><<<dim3(ceil(m/32.0),1,1),dim3(32,1,1)>>>(t, m, k, U, pitch_u);		
		// Get starting timepoint 
		auto start = std::chrono::high_resolution_clock::now(); 
		//Function to measure

		desal::cuda::DesalStatus res=desal::cuda::mg_vc_poisson<float, float2,std::ostream>(alpha, gamma,eta, 1, m,k, B, pitch_b, U, pitch_u,max_jacobi_iterations_per_stage,multigrid_stages, jacobi_weight, tol,early_termination_tol, &sos_residual);		
		// Get ending timepoint 
		auto stop = std::chrono::high_resolution_clock::now(); 
		
		auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start); 
	
		std::cout<<std::right<<"Time [ms]"<<std::setw(40)<<"V-Cycle Stages"<<std::setw(30)<<"Sum of Squares of Residual\n";

		std::cout<<std::right<<duration.count()<<std::setw(40)<<multigrid_stages<<std::setw(30)<<sos_residual<<"\n";
		if (res!=desal::cuda::DesalStatus::Success){
			if (res==desal::cuda::DesalStatus::CUDAError){
				std::cout<<"CudaError\n";
			}
			else if (res==desal::cuda::DesalStatus::InvalidParameters){
				std::cout<<"Invalid parameters supplied to the multigrid solver\n";
			}
			else{
				std::cout<<"Error could not successfully be reduced\n";
			}
			return;
		}		
		write_data_to_file(m,k, U, pitch_u, "output/", "output", iter);
		
		float2* temp=B;
		size_t pitch_temp=pitch_b;
		
		B=U;
		pitch_b=pitch_u;
		U=temp;
		pitch_u=pitch_temp;

		iter++;			
	}
	
	cudaFree(B);
	cudaFree(U);
}

int main(){
	size_t free_device_memory;
	size_t total_device_memory;
	cudaMemGetInfo(&free_device_memory,&total_device_memory);
	
	std::cout<<"This device has "<<total_device_memory/(1000000000.0)<<" Gigabytes of GPU memory\n";
	float height=3.0;
	float width=3.0;
	int m=1000;
	int k=1000;
	float interior_temperature=85;
	multigrid_example(interior_temperature,height,width,m,k,5);



}