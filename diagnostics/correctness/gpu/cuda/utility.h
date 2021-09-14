#pragma once 

__host__ 
void fill_array_ascendingly_f32(int n, float* arr, int stride,int offset=0);

__host__ 
void fill_array_descendingly_f32(int n, float* arr, int stride, int offset=0);

__host__ 
void fill_array_ascendingly_f64(int n, double* arr, int stride,int offset=0);
__host__
void fill_array_descendingly_f64(int n, double* arr, int stride, int offset=0);

__host__
void fill_array_ascendingly2D_f32(int m, int k, int boundary_padding_thickness, float2* U, int pitch_u, int offset=0);

template<class F2>
__global__
void k_fill_array_uniformly2D_field(int m,int k,int boundary_padding_thickness, F2* U,int pitch_u, F2 val, int offset=0){
	U=(F2*)((char*)U+boundary_padding_thickness*pitch_u);;
	for (int i=blockIdx.y*blockDim.y+threadIdx.y;i<(m-2*boundary_padding_thickness);i+=gridDim.y*blockDim.y){
		F2* temp=(F2*)((char*)U+i*pitch_u);
		for (int j=blockIdx.x*blockDim.x+threadIdx.x+boundary_padding_thickness;j<(k-boundary_padding_thickness);j+=gridDim.y*blockDim.x){
			temp[j]=val;
		}
	}
}

template<class F2>
__host__
void fill_array_uniformly2D(int m, int k, int boundary_padding_thickness, F2* U, int pitch_u, F2 val, int offset=0){
	float threads_x=32;
	float threads_y=32;
	dim3 threads=dim3(threads_x,threads_y,1);
	dim3 blocks=dim3(ceil(k/threads_x),ceil(m/threads_y),1);
	k_fill_array_uniformly2D_field<F2><<<blocks,threads>>>(m,k,boundary_padding_thickness,U,pitch_u,val, offset);	
}