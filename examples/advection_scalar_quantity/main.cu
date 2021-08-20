//#include<iostream>
#include<stdio.h>
#include "..\..\src\gpu\hostgpu_bindings.h"

__global__
void fill_matrices_k(int m,int k, float2* U, int pitch_u, float* Q, int pitch_q){
	int TILE_SIZE=4;
	int idux=(blockIdx.x*blockDim.x+threadIdx.x)*TILE_SIZE;
	int iduy=blockIdx.y*blockDim.y+threadIdx.y;
	int idqx=(blockIdx.x*blockDim.x+threadIdx.x)*TILE_SIZE;
	int idqy=blockIdx.y*blockDim.y+threadIdx.y;
	float* temp1=Q;
	
	for (int i=0;i<m;i+=gridDim.x*blockDim.x*TILE_SIZE){
		for (int j=0;j<k;j+=gridDim.y*blockDim.y){

		//	printf("%d ",idux);
			if (idqy<m && idqx <k){
				U=(float2*) ((char*)U+iduy*pitch_u);
				Q=(float*) ((char*)Q+idqy*pitch_q);
				for (int j=0;j<TILE_SIZE;j++){
					if (idux+j<k){
						//printf("%d ",idqx+j);
						U[idux+j].x=0;
						U[idux+j].y=1;
						Q[idqx+j]=idqy;		
					
					}
					else{
						return;
					}
				}			
			}
			else{
				return;
			}
			idux+=gridDim.x*blockDim.x;
			idqx+=gridDim.x*blockDim.x;
		}		
		iduy+=gridDim.y*blockDim.y;
		idqy+=gridDim.y*blockDim.y;
	}
	if (blockIdx.x==0 && threadIdx.y==0 && threadIdx.x==0 && blockIdx.y==0){
	temp1=(float*) ((char*)Q+3*pitch_q);
		temp1[3]=1.0;
	}

}

__global__
void print_matrix_k(int m,int k, float* M, int pitch,char name){
	printf("%c:\n",name);
	for (int i=0;i<m;i++){
		float* current_row=(float*)((char*)M + i*pitch);
		for (int j=0;j<k;j++){
			printf("%.1f  ",current_row[j]);
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

__host__
void run_example(float width, float height, int m, int k){
	float* Q; //quantity vector
	float2* U; //flow field vector
	float* C; // stores results of the advected quantity field Q
	
	float t0=0.0; //start time simulation
	float tend=10; //end time simulation
	float dt=1; //step size of the solver
	float dx=width/k;
	float dy=height/m;
	

	int sizeQ=sizeof(float)*m*k;
	int sizeU=2*sizeQ;
	size_t pitch_q;
	size_t pitch_u;
	
	//Allocate Device Memory	
	cudaMallocPitch((void**)&Q,&pitch_q,sizeof(float)*k,m);
	
	cudaMallocPitch((void**)&U,&pitch_u,sizeof(float2)*k,m);

	cudaMallocPitch((void**)&C,&pitch_q,sizeof(float)*k,m);
	dim3 threadLayout=dim3(32,32,1);
	dim3 blockLayout=dim3(2,2,1);
	fill_matrices_k<<<blockLayout,threadLayout>>>(m,k,U,pitch_u,Q,pitch_q);
	float t=t0+dt;
	print_matrix_k<<<1,1>>>(m,k,Q,pitch_q,'Q');
	//print_scalar_field(Q,n,m);	

	print_vector_field_k<<<1,1>>>(m,k,U,pitch_u,'U');

	while(t<=tend){

		advection_2D_f32_device(dt,1.0/dx,1.0/dy,m,k, U,pitch_u, Q,pitch_q, C, pitch_q);
		print_matrix_k<<<1,1>>>(m,k,C,pitch_q,'C');
		
		float* temp=Q;
		Q=C;
		C=temp;

		t+=dt;
	}
	
	cudaFree(Q);
	cudaFree(U);
	cudaFree(C);
}

int main(){
	float width=18.0; //width of the rectangular grid
	float height=18.0; //height of the rectangular grid
	float x_points=18; //number of gridpoints (including boundary points) in x direction 
	float y_points=18; //number of gridpoints (including boundary points) in y direction
	run_example(width,height,x_points,y_points);
		printf("Programmende\n");
}