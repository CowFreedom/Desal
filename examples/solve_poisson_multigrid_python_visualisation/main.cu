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
#include "../../src/gpu/cuda/solvers/navier_stokes.h"
#include "../../diagnostics/correctness/gpu/cuda/utility.h"

void write_data_to_file(int m, int k, float2* U_d, size_t pitch, std::string path, std::string name, int iter){
	std::ofstream file;
	file.open(path+"/"+name+std::to_string(iter)+".csv");
	float2* temp;
	
	temp=new float2[m*k];
	
	cudaMemcpy2D(temp,sizeof(float2)*k,U_d,pitch,k*sizeof(float2),m,cudaMemcpyDeviceToHost);
	
	for (int i=0;i<m;i++){
		for (int j=0;j<k-1;j++){
			float v1=temp[i*k+j].x;
			float v2=temp[i*k+j].y;
			float val=v1*v1+v2*v2;
			//std::cout<<v1<<"\t";
			file<<std::to_string(val)<<",";
			}
			float v1=temp[i*k+k-1].x;
			float v2=temp[i*k+k-1].y;
			float val=v1*v1+v2*v2;
			file<<std::to_string(val)<<"\n";
	}
	
	delete[] temp;
	file.close();
}

__device__
float expc(float a){

	if (a< -2.7){
		return 0.0;
	}
	else{
		return 1+a+0.5*a*a+(1.0/6)*a*a*a+(1.0/24)*a*a*a*a+(1.0/120)*a*a*a*a*a+(1.0/720)*a*a*a*a*a+(1.0/5040)*a*a*a*a*a*a*a+(1.0/40320)*a*a*a*a*a*a*a*a;
	}

}

template<class F, class F2>
__global__ 
void k_fill_force_blob(F dt, int boundary_padding, int m, int k, F2* U, int pitch){
	
	m-=2*boundary_padding;
	k-=2*boundary_padding;
	int idx=blockIdx.x*blockDim.x+threadIdx.x;
	int idy=blockIdx.y*blockDim.y+threadIdx.y;
	int xp=0.5*k;
	int yp=0.5*m;
	float inv_r=0.5;

	U=(F2*)((char*)U+(boundary_padding+idy)*pitch);
	for (int i=idy;i<m;i+=gridDim.y*blockDim.y){
		
		for (int j=idx;j<k;j+=gridDim.x*blockDim.x){
		
			float2 val;
			float temp=dt*expc(-inv_r*((j-xp)*(j-xp)+(i-yp)*(i-yp)));
			//printf("temp:%f für %d,%d und %f\n",temp, j, i, -inv_r*((j-xp)*(j-xp)+(i-yp)*(i-yp)));
			val.x=temp;
			val.y=0;
			U[j+boundary_padding]=val;
		}
		U=(F2*)((char*)U+pitch);
	}
}

template<class F2>
__global__
void k_boundary(float scale, int m, int k, F2* U, size_t pitch_u){
	int idx=blockIdx.x*blockDim.x+threadIdx.x;			
	if (idx>m || idx >k){
		return;
	}
	F2* U_neighbor=(F2*) ((char*)U+pitch_u);	
	F2 p;
	for (int i=idx;i<k;i+=gridDim.x*blockDim.x){
			p=U_neighbor[i];
			U[i].x=scale*p.x;
			U[i].y=scale*p.y;
	}	
	float2* U_ptr=(F2*) ((char*)U+(idx)*pitch_u);	
	for (int i=idx;i<m;i+=gridDim.x*blockDim.x){
		p=U_ptr[1];		
		U_ptr[0].x=scale*p.x;	
		U_ptr[0].y=scale*p.y;
		p=U_ptr[m-2];
		U_ptr[m-1].x=scale*p.x;
		U_ptr[m-1].y=scale*p.y;
		U_ptr=(F2*) ((char*)U_ptr+pitch_u);	
	}	
	U_neighbor=(F2*) ((char*)U+(m-2)*pitch_u);
	U_ptr=(F2*) ((char*)U_neighbor+pitch_u);	
	
	for (int i=idx;i<k;i+=gridDim.x*blockDim.x){
		p=U_neighbor[i];
		U_ptr[i].x=scale*p.x;
		U_ptr[i].y=scale*p.y;
		
	}
}

template<class F2>
__global__
void k_set_boundary_val(float2 val, int m, int k, F2* U, size_t pitch_u){
	int idx=blockIdx.x*blockDim.x+threadIdx.x;			

	F2* U_neighbor=(F2*) ((char*)U+pitch_u);	
	for (int i=idx;i<k;i+=gridDim.x*blockDim.x){
			U[i]=val;
	}	
	float2* U_ptr=(F2*) ((char*)U+(idx)*pitch_u);	
	for (int i=idx;i<m;i+=gridDim.x*blockDim.x){	
		U_ptr[0]=val;
		U_ptr[k-1]=val;
		U_ptr=(F2*) ((char*)U_ptr+pitch_u);	
	}	
	U_neighbor=(F2*) ((char*)U+(m-2)*pitch_u);
	U_ptr=(F2*) ((char*)U_neighbor+pitch_u);	
	
	for (int i=idx;i<k;i+=gridDim.x*blockDim.x){
		U_ptr[i]=val;	
	}
}

void multigrid_example(float height, float width, int m, int k, float tend){

	float dt=0.0125;
	float dx=width/k;
	float dy=height/m;
	int multigrid_stages=3;
	int max_jacobi_iterations_per_stage[]={50000,50000,40000,50000,10000,10000,10000,10000,2000,2000};//maximum number of iterations per multigrid stage
	float jacobi_weight=1; //weight factor of the weighted Jacobi method
	float tol=1e-1; //if sum of squares of the residual goes below this value, the estimation of x in Ax=b terminates and returns DesalStatus::Success
	float early_termination_tol=9.99e-1; //if residual_prev/residual_current is above early_termination_tol, then a Jacobi Iteration finishes early due to diminishing returns. If set to 1 then the iterations will not terminate early
	float sos_residual=-1; // this saves the sum of squares of the residual

	float v=0.425; //conductivity coefficient
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
	u_val.x=1;
	u_val.y=-4;
	
	float2 boundary_val;	
	boundary_val.x=1;
	boundary_val.y=3;
	
	cudaMemset2D(B,pitch_b,0.0,sizeof(float2)*k,m);	
	k_set_boundary_val<float2><<<dim3(ceil(m/32.0),1,1),dim3(32,1,1)>>>(boundary_val, m, k, B, pitch_b);
	cudaMemcpy2D(U,pitch_u,B,pitch_b,k*sizeof(float2),m,cudaMemcpyDeviceToHost);

	//desal::cuda::print_vector_field_k2<<<1,1>>>(m,k, B, pitch_b,'B');
	//desal::cuda::print_vector_field_k2<<<1,1>>>(m,k, U, pitch_u,'U');
	write_data_to_file(m, k, B, pitch_b, "output/", "output", 0);		
	int iter=1;

	for (float t=dt;t<=tend;t+=dt){	
		//The time taken to solve the linear system of equations is printed
		std::cout<<std::setprecision(3)<<std::fixed;
		
		// Get starting timepoint 
		auto start = std::chrono::high_resolution_clock::now(); 
		//Function to measure

		desal::cuda::DesalStatus res=desal::cuda::mg_vc_poisson_2D_device<float, float2,std::ostream>(alpha, gamma,eta, 1, m,k, B, pitch_b, U, pitch_u,max_jacobi_iterations_per_stage,multigrid_stages, jacobi_weight, tol,early_termination_tol, &sos_residual);		
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
	float height=1.0;
	float width=20.0;
	int m=150;
	int k=3000;
	multigrid_example(height,width,m,k,50);



}