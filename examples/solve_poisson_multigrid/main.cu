#include <stdio.h>

#include "..\..\src\gpu\hostgpu_bindings.h"

__global__
void test_correctness_k(float alpha, float beta, int m, int k, float2* A, float2* B, int pitch){

	float2* A_ptr=A;
	float2* B_ptr=B;
	A_ptr=(float2*) ((char*)A_ptr+pitch);
	B_ptr=(float2*) ((char*)B_ptr+pitch);	
	
	for (int i=0;i<m-2;i++){
		for (int j=1;j<k-1;j++){
			float2 aupper=*((float2*) ((char*)A_ptr+pitch));
			float2 alower=*((float2*) ((char*)A_ptr-pitch));
			float valx=(beta*A_ptr[j].x-(A_ptr[j-1].x+A_ptr[j+1].x+aupper.x+alower.x))/alpha;
			float valy=(beta*A_ptr[j].y-(A_ptr[j-1].y+A_ptr[j+1].y+aupper.y+alower.y))/alpha;
			if (valx!=B_ptr[j].x || valy!=B_ptr[j].y){
//printf("Error: valx:%f, vs. %f ; valy:%f, vs. %f",valx,B_ptr[j].x,valy,B_ptr[j].y);
				printf("Error:  valy:%f, vs. %f",valy,B_ptr[j].y);
			}
		}
		A_ptr=(float2*) ((char*)A_ptr+pitch);
		B_ptr=(float2*) ((char*)B_ptr+pitch);		
	}
}

__global__
void residual_k(float alpha, float beta, int m, int k, float2* A, float2* B, int pitch){

	float2* A_ptr=A;
	float2* B_ptr=B;
	A_ptr=(float2*) ((char*)A_ptr+pitch);
	B_ptr=(float2*) ((char*)B_ptr+pitch);	
	float sum=0;
	for (int i=0;i<m-2;i++){
		for (int j=1;j<k-1;j++){
			float2 aupper=*((float2*) ((char*)A_ptr+pitch));
			float2 alower=*((float2*) ((char*)A_ptr-pitch));
			float valx=(beta*A_ptr[j].x-(A_ptr[j-1].x+A_ptr[j+1].x+aupper.x+alower.x))/alpha;
			float valy=(beta*A_ptr[j].y-(A_ptr[j-1].y+A_ptr[j+1].y+aupper.y+alower.y))/alpha;
			sum+=(B_ptr[j].x-valx)*(B_ptr[j].x-valx)+(B_ptr[j].y-valy)+(B_ptr[j].y-valy);
		
		}
		A_ptr=(float2*) ((char*)A_ptr+pitch);
		B_ptr=(float2*) ((char*)B_ptr+pitch);		
	}
	printf("The residual is: %f\n",sum);
}

__global__
void fill_matrix_k(int m,int k, float2* U, int pitch_u){
	int TILE_SIZE=4;
	int idux=(blockIdx.x*blockDim.x+threadIdx.x)*TILE_SIZE;
	int iduy=blockIdx.y*blockDim.y+threadIdx.y;
	int idqx=(blockIdx.x*blockDim.x+threadIdx.x)*TILE_SIZE;
	int idqy=blockIdx.y*blockDim.y+threadIdx.y;
	float2* temp1=U;
	
	for (int i=0;i<m;i+=gridDim.x*blockDim.x*TILE_SIZE){
		for (int j=0;j<k;j+=gridDim.y*blockDim.y){

		//	printf("%d ",idux);
			if (idqy<m && idqx <k){
				U=(float2*) ((char*)U+iduy*pitch_u);
				for (int j=0;j<TILE_SIZE;j++){
					if (idux+j<k){
						//printf("%d ",idqx+j);
						U[idux+j].x=0;
						U[idux+j].y=1;
					
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
	temp1=(float2*) ((char*)U+3*pitch_u);
		temp1[3].x=7.0;
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
	float2* U; //flow field vector
	float2* C; // stores results of the advected quantity field Q
	

	float dt=1; //step size of the solver
	float dx=width/k;
	float dy=height/m;
	float v=1.0; //Viscousity coefficient
	
	size_t pitch_u;
	
	size_t pitch_c;
	
	//Allocate Device Memory	

	cudaMallocPitch((void**)&U,&pitch_u,sizeof(float2)*k,m);

	cudaMallocPitch((void**)&C,&pitch_c,sizeof(float2)*k,m);
	
	dim3 threadLayout=dim3(32,32,1);
	dim3 blockLayout=dim3(2,2,1);
	fill_matrix_k<<<blockLayout,threadLayout>>>(m,k,U,pitch_u);
	cudaMemcpy2D(C,pitch_c,U,pitch_u,k*sizeof(float2),m,cudaMemcpyDeviceToDevice);

	print_vector_field_k<<<1,1>>>(m,k,U,pitch_u,'U');

	float alpha=(dx*dy)/(v*dt);
	float beta=4.0+alpha;
	float2* C_ptr=(float2*) ((char*)C+pitch_c)+1;
	cudaMemset2D(C_ptr,pitch_c,0.0,(k-2)*sizeof(float2),m-2);	

	mg_vc_poisson_2D_f32(alpha, beta, m,k, U, pitch_u, C, pitch_c); //only use interior points
	
	//test_correctness_k<<<1,1>>>(alpha,beta,m,k,C,U,pitch_u);
	residual_k<<<1,1>>>(alpha,beta,m,k,C,U,pitch_u);
	print_vector_field_k<<<1,1>>>(m,k,C,pitch_c,'C'); 
	cudaFree(U);
	cudaFree(C);
}

int main(){
	float width=18.0; //width of the rectangular grid
	float height=18.0; //height of the rectangular grid
	float x_points=18; //number of gridpoints (including boundary points) in x direction 
	float y_points=18; //number of gridpoints (including boundary points) in y direction
	run_example(width,height,x_points,y_points);
}