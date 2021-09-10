#include "utility.h"
#include "../../../../src/gpu/cuda/reductions.h"
#include <stdio.h>



__global__
void print_matrix_k(int m,int k, float* M, int stride_col,int stride_row,char name){
	printf("%c:\n",name);
	for (int i=0;i<m;i++){
		for (int j=0;j<k;j++){
			printf("%.1f ",M[i*stride_col+j*stride_row]);
		}
		printf("\n");
	}	
}

bool test_reduce_sum_f32_device_ascending(int n, int reps, char* error_message=nullptr){

	for (int i=1;i<=reps;i++){
		n=i*n;
		float result=-1;
		long long int temp=n;
		float sum=(temp*(temp+1))*0.5;
		float* v;
		int size_v=sizeof(float)*n;
		cudaMalloc((void**) &v,size_v);
		fill_array_ascendingly_f32(n,v,1,0);
		reduce_sum_f32_device(n,v,1);	
		cudaMemcpy(&result,v,sizeof(float)*1,cudaMemcpyDeviceToHost);
		if (result!=sum){
		
			
			cudaFree(v);
			return false;
		}
		cudaFree(v);
	}

	return true;
}

bool test_reduce_sum_f32_device_descending(int n, int reps, char* error_message=nullptr){

	for (int i=1;i<=reps;i++){
		n=i*n;
		float result=-1;
		long long int temp=n;
		float sum=(temp*(temp+1))*0.5;
		float* v;
		int size_v=sizeof(float)*n;
		cudaMalloc((void**) &v,size_v);
		fill_array_descendingly_f32(n,v,1,0);
		reduce_sum_f32_device(n,v,1);	
		cudaMemcpy(&result,v,sizeof(float)*1,cudaMemcpyDeviceToHost);
		if (result!=sum){
			cudaFree(v);
			return false;
		}
		cudaFree(v);
	}

	return true;
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

bool test_reduce_sum_of_squares_poisson_field_residual_f32_device_ascending(int n, int reps, char* error_message=nullptr){

	for (int i=1;i<=1;i++){	
		float2* U; //flow field vector
		float2* B; //flow field vector
		float* r; // stores diffused velocity field
		int n_mines_one_squared=(n-1)*(n-1); //Sum of squares is taken of interior points
		int result=(n_mines_one_squared*(n_mines_one_squared+1)*(2*n_mines_one_squared+1))/6; //
		int width=n;
		int height=n;
		int k=n;
		int m=n;
		
		float dt=1;
		float dx=width/k;
		float dy=height/m;
		
		float v=1.0; //Viscousity coefficient
		float alpha=(dx*dy)/(v*dt);
		float beta=4.0+alpha;	
		
		size_t pitch_u;
		size_t pitch_b;
		
		//Allocate Device Memory	
		cudaMallocPitch((void**)&U,&pitch_u,sizeof(float2)*k,m);
		cudaMallocPitch((void**)&B,&pitch_b,sizeof(float2)*k,m);

		cudaMemcpy2D(B,pitch_b,U,pitch_u,k*sizeof(float2),m,cudaMemcpyDeviceToDevice);
		cudaMemset2D(B,pitch_b,0.0,(k-2)*sizeof(float2),m-2);	

		cudaMalloc((void**)&r,sizeof(float)*k*m);
		cudaMemset(r,0.0,sizeof(float)*k*m);
		
		
		dim3 threadLayout=dim3(32,32,1);
		dim3 blockLayout=dim3(2,2,1);
		fill_array_ascendingly2D_f32(m,k,1,U,pitch_u);
		
		
		reduce_sum_of_squares_poisson_field_residual_f32_device(alpha,beta,1, n,U,pitch_u, B, pitch_b, r, 1);
		print_vector_field_k<<<1,1>>>(m,k,U, pitch_u,'U');
		//print_vector_field_k<<<1,1>>>(m,k,B, pitch_b,'B');
		print_matrix_k<<<1,1>>>(1,m*k, r, m,1,'r');		
		printf("Result is:%d\n",result);

		cudaFree(U);
		cudaFree(r);		
	}

	return true;
}

bool test_reduce_sum_of_squares_poisson_field_residual_f32_device_uniform(int n, int reps, char* error_message=nullptr){

	for (int i=1;i<=reps;i++){	
		n=i*n;
		float2* U; //flow field vector
		float2* B; //flow field vector
		float* r; // stores diffused velocity field
		int width=n;
		int height=n;
		int k=n;
		int m=n;
		
		float dt=1;
		float dx=width/k;
		float dy=height/m;
		
		float v=1.0; //Viscousity coefficient
		float alpha=(dx*dy)/(v*dt);
		float beta=4.0+alpha;	
		
		size_t pitch_u;
		size_t pitch_b;
		
		//Allocate Device Memory	
		cudaMallocPitch((void**)&U,&pitch_u,sizeof(float2)*k,m);
		cudaMallocPitch((void**)&B,&pitch_b,sizeof(float2)*k,m);

		cudaMemcpy2D(B,pitch_b,U,pitch_u,k*sizeof(float2),m,cudaMemcpyDeviceToDevice);
		cudaMemset2D(B,pitch_b,0.0,(k-2)*sizeof(float2),m-2);	

		cudaMalloc((void**)&r,sizeof(float)*k*m);
		cudaMemset(r,0.0,sizeof(float)*k*m);
		
		float2 u_val;
		float2 b_val;
		b_val.x=1;
		b_val.y=1;
		u_val.x=13;
		u_val.y=1;
		fill_array_uniformly2D<float2>(m,k,1,U,pitch_u,u_val);
		fill_array_uniformly2D<float2>(m,k,1,B,pitch_b,b_val);
		
		
		float2 v_inner;
		float2 v_margins;
		float2 v_corners;
		v_inner.x=(beta*u_val.x-4*u_val.x)/alpha;
		v_inner.y=(beta*u_val.y-4*u_val.y)/alpha;
		v_margins.x=((beta-3)*u_val.x)/alpha;
		v_margins.y=((beta-3)*u_val.y)/alpha;
		v_corners.x=((beta-2)*u_val.x)/alpha;
		v_corners.y=((beta-2)*u_val.y)/alpha;
		//printf("vi:%f, vm: %f, vc%f\n",v_inner.x,v_margins.x,v_corners.x);
		float exact_result=(n-4)*(n-4)*((b_val.x-v_inner.x)*(b_val.x-v_inner.x)+(b_val.y-v_inner.y)*(b_val.y-v_inner.y));
		exact_result+=4*(n-4)*((b_val.x-v_margins.x)*(b_val.x-v_margins.x)+(b_val.y-v_margins.y)*(b_val.y-v_margins.y));
		exact_result+=4*((b_val.x-v_corners.x)*(b_val.x-v_corners.x)+(b_val.y-v_corners.y)*(b_val.y-v_corners.y));
		
		
		reduce_sum_of_squares_poisson_field_residual_f32_device(alpha,beta,1, n,U,pitch_u, B, pitch_b, r, 1);
		//print_vector_field_k<<<1,1>>>(m,k,U, pitch_u,'U');
		//print_vector_field_k<<<1,1>>>(m,k,B, pitch_b,'B');
	//	print_matrix_k<<<1,1>>>(1,m*k, r, m,1,'r');		
	//	printf("Result is:%f\n",exact_result);
		
		float gpu_result;
		cudaMemcpy(&gpu_result,r,sizeof(float)*1,cudaMemcpyDeviceToHost);
		
		cudaFree(r);
		cudaFree(U);
	//	printf("Exact result is:%f Gpu result is: %f\n",exact_result, gpu_result);
		if (gpu_result!=exact_result){
			printf("Exact result is:%f Gpu result is: %f\n",exact_result, gpu_result);
			return false;
		}
			
	}

	return true;
}


bool test_reduce_sum_f64_device_ascending(int n, int reps, char* error_message=nullptr){

	for (int i=1;i<=reps;i++){
		n=i*n;
		double result=-1;
		long long int temp=n;
		double sum=(temp*(temp+1))*0.5;
		double* v;
		int size_v=sizeof(double)*n;
		cudaMalloc((void**) &v,size_v);
		fill_array_ascendingly_f64(n,v,1,0);
		reduce_sum_f64_device(n,v,1);	
		cudaMemcpy(&result,v,sizeof(double)*1,cudaMemcpyDeviceToHost);
		if (result!=sum){			
			cudaFree(v);
			return false;
		}
		cudaFree(v);
	}

	return true;
}

bool test_reduce_sum_f64_device_descending(int n, int reps, char* error_message=nullptr){

	for (int i=1;i<=reps;i++){
		n=i*n;
		double result=-1;
		long long int temp=n;
		double sum=(temp*(temp+1))*0.5;
		double* v;
		int size_v=sizeof(double)*n;
		cudaMalloc((void**) &v,size_v);
		fill_array_descendingly_f64(n,v,1,0);
		reduce_sum_f64_device(n,v,1);	
		cudaMemcpy(&result,v,sizeof(double)*1,cudaMemcpyDeviceToHost);
		if (result!=sum){
			cudaFree(v);
			return false;
		}
		cudaFree(v);
	}

	return true;
}