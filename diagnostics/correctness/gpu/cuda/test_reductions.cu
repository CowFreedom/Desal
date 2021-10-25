#include "utility.h"
#include "../../../../src/gpu/cuda/reductions.h"
#include <stdio.h>

namespace desal{
	
	namespace cuda{

		__global__
		void print_vector_field_k3(int m,int k, float2* M, int pitch,char name){
			printf("%c:\n",name);
			for (int i=0;i<m;i++){
				float2* current_row=(float2*)((char*)M + i*pitch);
				for (int j=0;j<k;j++){
					printf("(%.1f,%.1f) ",current_row[j].x,current_row[j].y);
				}
				printf("\n");
			}	
		}
/*
		*/
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
				reduce_sum_device<float>(n,v,1);	
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
				reduce_sum_device<float>(n,v,1);	
				cudaMemcpy(&result,v,sizeof(float)*1,cudaMemcpyDeviceToHost);
				if (result!=sum){
					cudaFree(v);
					return false;
				}
				cudaFree(v);
			}

			return true;
		}

		bool test_reduce_sum_of_squares_poisson_field_residual_f32_device_uniform(int reps, char* error_message=nullptr){

			for (int i=0;i<reps;i++){	
				float2* U; //flow field vector
				float2* B; //flow field vector
				float* r; // stores diffused velocity field
				int k=1<<(2+i);
				int m=1<<(2+i);
				
				float dt=1;
				float dx=1;
				float dy=1;
				
				float v=1.0; //Viscousity coefficient
				float alpha=(dx*dy)/(v*dt);
				float beta=4.0+alpha;	
				
				size_t pitch_u;
				size_t pitch_b;
				
				//Allocate Device Memory	
				cudaMallocPitch((void**)&U,&pitch_u,sizeof(float2)*k,m);
				cudaMallocPitch((void**)&B,&pitch_b,sizeof(float2)*k,m);

				cudaMemset2D(B,pitch_b,0.0,k*sizeof(float2),m);	
				cudaMemset2D(U,pitch_u,0.0,k*sizeof(float2),m);	
		
				int blocks_x=ceil(k/(2.0*desal::cuda::reductions::blocksizes::a::MX));
				int blocks_y=ceil(m/(2.0*desal::cuda::reductions::blocksizes::a::MY));
			
				cudaMalloc((void**)&r,sizeof(float)*k*m);
				cudaMemset(r,0.0,sizeof(float)*k*m);
				
				float2 u_val;
				float2 b_val;
				b_val.x=1;
				b_val.y=1;
				u_val.x=13;
				u_val.y=1;
				
				float2 boundary_val;
				boundary_val.x=1;
				boundary_val.y=-3;
							
				fill_array_uniformly2D<float2>(m,k,1,U,pitch_u,u_val);
				k_set_boundary_val<float2><<<dim3(ceil(m/32.0),1,1),dim3(32,1,1)>>>(boundary_val, m, k, U, pitch_u);			
				fill_array_uniformly2D<float2>(m,k,1,B,pitch_b,b_val);	

				float2 v_inner;
				float2 v_margins;
				float2 v_corners;
				v_inner.x=(beta*u_val.x-4*u_val.x)/alpha;
				v_inner.y=(beta*u_val.y-4*u_val.y)/alpha;
				v_margins.x=((beta-3)*u_val.x-boundary_val.x)/alpha;
				v_margins.y=((beta-3)*u_val.y-boundary_val.y)/alpha;
				v_corners.x=((beta-2)*u_val.x-2*boundary_val.x)/alpha;
				v_corners.y=((beta-2)*u_val.y-2*boundary_val.y)/alpha;
				//printf("vi:%f, vm: %f, vc%f\n",v_inner.x,v_margins.x,v_corners.x);
				float exact_result=(m-4)*(k-4)*((b_val.x-v_inner.x)*(b_val.x-v_inner.x)+(b_val.y-v_inner.y)*(b_val.y-v_inner.y));
				exact_result+=2*((m-4)+(k-4))*((b_val.x-v_margins.x)*(b_val.x-v_margins.x)+(b_val.y-v_margins.y)*(b_val.y-v_margins.y));
				exact_result+=4*((b_val.x-v_corners.x)*(b_val.x-v_corners.x)+(b_val.y-v_corners.y)*(b_val.y-v_corners.y));
				
				reduce_sum_of_squares_poisson_field_residual_device<float,float2>(alpha,beta,1, m,k,U,pitch_u, B, pitch_b, r, 1);
			
				float gpu_result;
				cudaMemcpy(&gpu_result,r,sizeof(float)*1,cudaMemcpyDeviceToHost);
				
				cudaFree(r);
				cudaFree(U);
				cudaFree(B);
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
				reduce_sum_device<double>(n,v,1);	
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
				reduce_sum_device<double>(n,v,1);	
				cudaMemcpy(&result,v,sizeof(double)*1,cudaMemcpyDeviceToHost);
				if (result!=sum){
					cudaFree(v);
					return false;
				}
				cudaFree(v);
			}

			return true;
		}
	}
}