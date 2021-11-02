/** \file main.cpp
 * This example calculates u(t+1,x)=u(t,x-dt*u(t,x))
 */
 
#include <iostream>
#include <iomanip>
#include <chrono>
#include <fstream>
#include <string>
#include <filesystem>
#include "../../src/gpu/cuda/error_handling.h"
#include "../../src/gpu/cuda/solvers/advection.h"
#include "../../diagnostics/correctness/gpu/cuda/utility.h"

void write_data_to_file(int m, int k, float2* U_d, size_t pitch, std::string path, std::string name, int iter){
	std::ofstream file;
	std::filesystem::create_directory("output");
	file.open(path+"/"+name+std::to_string(iter)+".csv");
	float2* temp;
	
	temp=new float2[m*k];
	
	cudaMemcpy2D(temp,sizeof(float2)*k,U_d,pitch,k*sizeof(float2),m,cudaMemcpyDeviceToHost);
	
	for (int i=0;i<m;i++){
		for (int j=0;j<k-1;j++){
			float v1=temp[i*k+j].x;
			float v2=temp[i*k+j].y;
			float val=v1*v1+v2*v2;
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

//Definition of the wave like initial conditions
template<class F, class F2>
__global__ 
void k_wave(F height, F width, F dt, int boundary_padding, int m, int k, F2* U, int pitch){
	
	m-=2*boundary_padding;
	k-=2*boundary_padding;
	int idx=blockIdx.x*blockDim.x+threadIdx.x;
	int idy=blockIdx.y*blockDim.y+threadIdx.y;
	int xp=0.2*width;
	int yp=0.2*m;
	float inv_r=0.1;

	U=(F2*)((char*)U+(boundary_padding+idy)*pitch);
	for (int i=idy;i<m;i+=gridDim.y*blockDim.y){
		
		for (int j=idx;j<k;j+=gridDim.x*blockDim.x){
		
			float2 val;
			float temp=expf(-inv_r*((static_cast<F>(j)/k)*width-xp)*((static_cast<F>(j)/k)*width-xp));
			val.x=temp;
			val.y=0;
			U[j+boundary_padding]=val;
		}
		U=(F2*)((char*)U+pitch);
	}
}

void multigrid_example(float height, float width, int m, int k, float tend){

	float dt=0.125;
	float dx=width/k;
	float dy=height/m;
	
	//Allocate Vectors
	float2* B; //flow field vector
	float2* U; //flow field vector buffer	

	size_t pitch_b;
	size_t pitch_u;
	
	cudaMallocPitch((void**)&U,&pitch_u,sizeof(float2)*k,m);
	cudaMallocPitch((void**)&B,&pitch_b,sizeof(float2)*k,m);	
	
	//Initialize grid
	
	int boundary_padding=0;
	cudaMemset2D(B,pitch_b,0.0,sizeof(float2)*k,m);	

	k_wave<<<dim3(ceil(m/32.0),ceil(k/32.0),1),dim3(32,32,1)>>>(height,width,dt,boundary_padding,m,k,U,pitch_u);	

	write_data_to_file(m,k, U, pitch_u, "output/", "output", 0);
		
	int iter=1;

	for (float t=dt;t<=tend;t+=dt){	

		std::cout<<std::setprecision(3)<<std::fixed;
		
		// Get starting timepoint 
		auto start = std::chrono::high_resolution_clock::now(); 
		
		cudaError_t res=desal::cuda::advection(dt, boundary_padding,dy,dx,m,k,U, pitch_u, U,pitch_u, B,pitch_b);		
		
		// Get ending timepoint 
		auto stop = std::chrono::high_resolution_clock::now(); 
		
		auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start); 
	
		std::cout<<"Percent done:"<<(t/tend)*100<<"\tTime taken this iteration [microseconds]:"<<duration.count()<<"\n";
		if (res!= cudaSuccess){
			std::cout<<cudaGetErrorString(res);
			return;
		}

		write_data_to_file(m, k, B, pitch_b, "output/", "output", iter);			
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
	
	std::cout<<"The utilized CUDA device has "<<total_device_memory/(1000000000.0)<<" Gigabytes of memory\n";
	float height=1.0;
	float width=10.0;
	int m=200;
	int k=2000;
	multigrid_example(height,width,m,k,100);
}