#pragma once
#include "../error_handling.h"
#include "../reductions.h"
#include "../transformations.h"
/*
template<class S>
DesalStatus mg_vc_poisson_2D_f32_device(float alpha, float gamma, float eta, int boundary_padding_thickness, int n, float2* B_d, int pitch_b, float2* C_d, int pitch_c, S* os=nullptr);
*/

namespace desal {
	
	namespace cuda{
	
		template<class F, class F2>
		__global__
		void k_jacobi_poisson_2D(F weight, F alpha, F beta_inv, int boundary_padding_thickness, int n, cudaTextureObject_t X_old, F2* X_new, int pitch_x, cudaTextureObject_t B);

		__global__
		void print_vector_field_k2(int m,int k, float2* M, int pitch,char name, int iteration=0);

		//Solves AX=B
		template<class F, class F2, class S>
		__host__
		DesalStatus jacobi_poisson_2D_device(F jacobi_weight, F alpha, F beta, int boundary_padding_thickness, int n, F2* X, int pitch_x, F2* X_buf, int pitch_x_buf, cudaTextureObject_t B, int jacobi_iterations, float2* B_d, int pitch_b, S* os=nullptr){
			F beta_inv=1.0/beta;
			//Create Resource description
			cudaResourceDesc resDescX_buf;
			cudaResourceDesc resDescX;
			memset(&resDescX_buf,0,sizeof(resDescX_buf));
			memset(&resDescX,0,sizeof(resDescX));

			resDescX_buf.resType = cudaResourceTypePitch2D;
			resDescX_buf.res.pitch2D.devPtr=X_buf;
			resDescX_buf.res.pitch2D.pitchInBytes=pitch_x_buf;
			resDescX_buf.res.pitch2D.width=n;
			resDescX_buf.res.pitch2D.height=n;
			resDescX_buf.res.pitch2D.desc=cudaCreateChannelDesc<F2>(); //cudaCreateChannelDesc(32,32,0,0,cudaChannelFormatKindFloat) is equivalent cudaCreateChannelDesc<float2>()

			resDescX.resType = cudaResourceTypePitch2D;
			resDescX.res.pitch2D.devPtr=X;
			resDescX.res.pitch2D.pitchInBytes=pitch_x;
			resDescX.res.pitch2D.width=n;
			resDescX.res.pitch2D.height=n;
			resDescX.res.pitch2D.desc=cudaCreateChannelDesc<F2>(); //cudaCreateChannelDesc(32,32,0,0,cudaChannelFormatKindFloat) is equivalent cudaCreateChannelDesc<float2>()


			//Create Texture description
			cudaTextureDesc texDesc;
			memset(&texDesc,0,sizeof(texDesc));
			texDesc.normalizedCoords = false;
			texDesc.filterMode = cudaFilterModeLinear; //change to nearest
			texDesc.readMode=cudaReadModeElementType;
			texDesc.addressMode[0] = cudaAddressModeBorder;
			texDesc.addressMode[1] = cudaAddressModeBorder;
			
			cudaTextureObject_t X_buf_tex;
			
			cudaTextureObject_t X_tex;

			cudaError_t err=cudaCreateTextureObject(&X_buf_tex, &resDescX_buf, &texDesc, NULL);
			if (err != cudaSuccess){
					return DesalStatus::CUDAError;
			}
				
			err=cudaCreateTextureObject(&X_tex, &resDescX, &texDesc, NULL);
			if (err != cudaSuccess){
					return DesalStatus::CUDAError;
			}
		//print_vector_field_k2<<<1,1>>>(n,n,B_d,pitch_b,'B');

			int threads_per_block_x=8;	
			int threads_per_block_y=4;	
			int blocks_x=ceil(static_cast<float>(n)/(threads_per_block_x));
			int blocks_y=ceil(static_cast<float>(n)/(threads_per_block_y));
			
			dim3 threads=dim3(threads_per_block_x,threads_per_block_y,1);
			dim3 blocks=dim3(blocks_x,blocks_y,1);
			int iteration_blocks=jacobi_iterations/2;
			for (int i=0;i<iteration_blocks;i++){
			//	print_vector_field_k2<<<1,1>>>(n,n,X_old,pitch_x_old,'O',i);

				F* test;
				
				cudaMalloc((void**) &test, sizeof(F)*blocks_x*blocks_y);
				cudaError_t err=gpuErrorCheck(cudaMemset(test,0,sizeof(F)*blocks_x*blocks_y),os);
				if (err !=cudaSuccess){
					return DesalStatus::CUDAError;
				}
				reduce_sum_of_squares_poisson_field_residual_f32_device(alpha,beta,boundary_padding_thickness, n,X,pitch_x, B_d, pitch_b, test, 1);	
			//	print_vector_k2<<<1,1>>>(10, test,'r', i);
				F residual;
				cudaMemcpy(&residual,test,sizeof(F)*1,cudaMemcpyDeviceToHost);
				printf("mg vc poisson 2d residual: %.12f round: %d\n",residual, i);
				cudaFree(test);
				
			//	printf("S1: a%f b %f\n",alpha,beta_inv);
				k_jacobi_poisson_2D<F,F2><<<blocks,threads>>>(jacobi_weight,alpha,beta_inv,boundary_padding_thickness,n,X_tex,X_buf,pitch_x_buf,B);	

			//	print_vector_field_k2<<<1,1>>>(n,n,X,pitch_x,'X');


				k_jacobi_poisson_2D<F,F2><<<blocks,threads>>>(jacobi_weight,alpha,beta_inv,boundary_padding_thickness,n,X_buf_tex,X,pitch_x_buf,B);			
			
				//print_vector_field_k2<<<blocks,threads>>>(n,n,X_old,pitch_x_old,'X');
			}
			//TODO: No need to always keep both buffers synchronized
			if (jacobi_iterations%2 ==0){
				
				F* test;
				
				cudaMalloc((void**) &test, sizeof(F)*blocks_x*blocks_y);
				err=gpuErrorCheck(cudaMemset(test,0,sizeof(F)*blocks_x*blocks_y),os);
				if (err != cudaSuccess){
					return DesalStatus::CUDAError;
				}
				reduce_sum_of_squares_poisson_field_residual_f32_device(alpha,beta,boundary_padding_thickness, n,X,pitch_x, B_d, pitch_b, test, 1);	
			//	print_vector_k2<<<1,1>>>(10, test,'r', i);
				F residual;
				cudaMemcpy(&residual,test,sizeof(F)*1,cudaMemcpyDeviceToHost);
				printf("mg vc poisson 2d residual: %.12f\n",residual);
				cudaFree(test);
			}
			else{
			
				k_jacobi_poisson_2D<F,F2><<<blocks,threads>>>(jacobi_weight,alpha,beta_inv,boundary_padding_thickness,n,X_tex,X_buf,pitch_x_buf,B);		
				err=gpuErrorCheck(cudaMemcpy2D(X,pitch_x,X_buf,pitch_x_buf,n*sizeof(F2),n,cudaMemcpyDeviceToDevice),os);
				
				if (err != cudaSuccess){
					return DesalStatus::CUDAError;
				}
				//printf("n:%d\n",n);
				//print_vector_field_k2<<<1,1>>>(n,n,X,pitch_x,'Q');
				
				F* test;
				
				cudaMalloc((void**) &test, sizeof(F)*blocks_x*blocks_y);
				cudaError_t err=gpuErrorCheck(cudaMemset(test,0,sizeof(F)*blocks_x*blocks_y),os);
				
				if (err != cudaSuccess){
					return DesalStatus::CUDAError;
				}
			//	print_vector_k2<<<1,1>>>(blocks_x*blocks_y, test,'l');
				reduce_sum_of_squares_poisson_field_residual_f32_device(alpha,beta,boundary_padding_thickness, n,X,pitch_x, B_d, pitch_b, test, 1);	
				//print_vector_k2<<<1,1>>>(blocks_x*blocks_y, test,'r');
				F residual;
				cudaMemcpy(&residual,test,sizeof(F)*1,cudaMemcpyDeviceToHost);
				printf("mg vc poisson 2d residual uneven: %.12f\n",residual);
				cudaFree(test);
			}
			return DesalStatus::Success;
		}


		//AC=B
		template<class F, class F2, class S>
		__host__
		DesalStatus mg_vc_poisson_2D_nobuf_device(F alpha, F gamma, F eta, int boundary_padding_thickness, int n, F2* B_d, int pitch_b, F2* C_d, int pitch_c, F2* C_buf, int pitch_c_buf, F2* r_buf, int pitch_r_buf, F jacobi_weight, int jacobi_iterations, int multigrid_stages, S* os){
			
			F beta=gamma+eta;
			//Create Resource description
			cudaResourceDesc resDesc;
			memset(&resDesc,0,sizeof(resDesc));
			resDesc.resType = cudaResourceTypePitch2D;
			resDesc.res.pitch2D.devPtr=B_d;
			resDesc.res.pitch2D.width=n;
			resDesc.res.pitch2D.height=n;
			resDesc.res.pitch2D.pitchInBytes=pitch_b;
			resDesc.res.pitch2D.desc=cudaCreateChannelDesc<F2>(); 

			//Create Texture description
			cudaTextureDesc texDesc;
			memset(&texDesc,0,sizeof(texDesc));
			texDesc.normalizedCoords = false;
			texDesc.filterMode = cudaFilterModeLinear;
			texDesc.readMode=cudaReadModeElementType;
			texDesc.addressMode[0] = cudaAddressModeClamp;
			texDesc.addressMode[1] = cudaAddressModeClamp;

			//Create Texture Object
			cudaTextureObject_t B_tex;
			cudaError_t err=gpuErrorCheck(cudaCreateTextureObject(&B_tex, &resDesc, &texDesc, NULL),os);
			if (err !=cudaSuccess){
				return DesalStatus::CUDAError;
			}

			jacobi_poisson_2D_device<F,F2,S>(jacobi_weight,alpha,beta,boundary_padding_thickness,n, C_d,pitch_c,C_buf,pitch_c_buf, B_tex,jacobi_iterations,B_d,pitch_b,os);

			transform_entries_into_residuals_device<float,float2>(alpha,beta, boundary_padding_thickness, n,n, C_buf, pitch_c_buf, B_d, pitch_b, r_buf, pitch_r_buf); //TODO C_buf should be equal to C at this stage
	
			int ns[20];
			ns[0]=n;
			ns[1]=restrict_n(n);
			F2* C_buf_prev=C_buf;
			F2* r_buf_prev=r_buf;
			F alpha_curr=alpha;
			F beta_curr;
			F gamma_curr=gamma;
			F2* C_buf_curr=(F2*)((char*)C_buf+n*pitch_c_buf);	
			F2* r_buf_curr=(F2*)((char*)r_buf+n*pitch_r_buf);
			F2* r_buf_next=(F2*)((char*)r_buf_curr+ns[1]*pitch_r_buf);
			printf("Down Cycle\n");
			
			/*Synchronize Buffer C_buf to contain the same values as C_d*/
			if ((jacobi_iterations%2) ==0){
				err=gpuErrorCheck(cudaMemcpy2D(C_buf,pitch_c_buf,C_d,pitch_c,n*sizeof(float2),n,cudaMemcpyDeviceToDevice),os);
				if (err != cudaSuccess){
					return DesalStatus::CUDAError;
				}
			}	
			for (int stage=1;stage<multigrid_stages;stage++){
				
				//Restrict previous residuals
				restrict<F,F2>(ns[stage-1],ns[stage], r_buf_curr, pitch_r_buf, r_buf_prev, pitch_r_buf);

				//cudaDeviceSynchronize();
				//Solve new system Ax=r
				cudaResourceDesc resDescR;
				memset(&resDescR,0,sizeof(resDescR));
				resDescR.resType = cudaResourceTypePitch2D;
				resDescR.res.pitch2D.devPtr=r_buf_curr;
				resDescR.res.pitch2D.width=ns[stage];
				resDescR.res.pitch2D.height=ns[stage];
				resDescR.res.pitch2D.pitchInBytes=pitch_r_buf;
				resDescR.res.pitch2D.desc=cudaCreateChannelDesc<F2>(); 		
				
				cudaTextureObject_t r_tex;
				err=gpuErrorCheck(cudaCreateTextureObject(&r_tex, &resDescR, &texDesc, NULL),os);
				if (err!=cudaSuccess){
					return DesalStatus::CUDAError;
				}
				alpha_curr=0.25*alpha;
				gamma_curr=0.25*gamma;
				beta_curr=gamma_curr+eta;
				
				err=gpuErrorCheck(cudaMemcpy2D(C_d,pitch_c,C_buf_curr,pitch_c_buf,ns[stage]*sizeof(float2),ns[stage],cudaMemcpyDeviceToDevice),os);
				if (err!=cudaSuccess){
					return DesalStatus::CUDAError;
				}
				
				printf("ns[stage]:%d\n",ns[stage]);

				jacobi_poisson_2D_device<F,F2,S>(jacobi_weight,alpha_curr,beta_curr,boundary_padding_thickness,ns[stage], C_buf_curr,pitch_c_buf,C_d,pitch_c, r_tex,jacobi_iterations,r_buf_curr,pitch_r_buf,os);

				if (stage<(multigrid_stages-1)){
					transform_entries_into_residuals_device<float,float2>(alpha_curr,beta_curr, boundary_padding_thickness, ns[stage],ns[stage], C_buf_curr, pitch_c_buf, r_buf_curr, pitch_b, r_buf, pitch_r_buf); //TODO C_buf should be equal to C at this stage

					//Iterate pointers
					C_buf_prev=C_buf_curr;
					r_buf_prev=r_buf_curr;
					C_buf_curr=(F2*)((char*)C_buf_curr+ns[stage]*pitch_c_buf);	
					r_buf_curr=r_buf_next;
					ns[stage+1]=restrict_n(ns[stage]);//(ns-1)*0.5+1; //there are ns-1 spaces between nodes
					r_buf_next=(F2*)((char*)r_buf_curr+ns[stage+1]*pitch_r_buf);
				}
				
			}


			printf("Up Cycle\n");
			for (int stage=multigrid_stages;stage>1;stage--){
				alpha_curr=4*alpha;
				gamma_curr=4*gamma;
				beta_curr=gamma_curr+eta;
				F2* C_buf_next=(F2*)((char*)C_buf_curr-ns[stage-2]*pitch_c_buf);
				printf("ns next: %d ns: %d\n",ns[stage-2],ns[stage-1]);

				//prolongate and correct previous result
				
				prolong_and_add(ns[stage-2],ns[stage-1],C_buf_next,pitch_c_buf,C_buf_curr,pitch_c_buf);

				k_check_boundary<<<1,1>>>(ns[stage-2],C_buf_next, pitch_c_buf,0.0);
				
				//relax again
				err=gpuErrorCheck(cudaMemcpy2D(C_d,pitch_c,C_buf_next,pitch_c_buf,ns[stage-2]*sizeof(float2),ns[stage-2],cudaMemcpyDeviceToDevice),os);
				
				if (err!=cudaSuccess){
					return DesalStatus::CUDAError;
				}
				
				if (stage>2){
					r_buf_next=(F2*)((char*)r_buf_curr-ns[stage-2]*pitch_r_buf);		
					cudaResourceDesc resDescR;
					memset(&resDescR,0,sizeof(resDescR));
					resDescR.resType = cudaResourceTypePitch2D;
					resDescR.res.pitch2D.devPtr=r_buf_next;
					resDescR.res.pitch2D.width=ns[stage-2];
					resDescR.res.pitch2D.height=ns[stage-2];
					resDescR.res.pitch2D.pitchInBytes=pitch_r_buf;
					resDescR.res.pitch2D.desc=cudaCreateChannelDesc<F2>(); 		
					
					cudaTextureObject_t r_tex;
					err=gpuErrorCheck(cudaCreateTextureObject(&r_tex, &resDescR, &texDesc, NULL),os);

					if (err!=cudaSuccess){
						return DesalStatus::CUDAError;
					}
					jacobi_poisson_2D_device<F,F2,S>(jacobi_weight,alpha_curr,beta_curr,boundary_padding_thickness,ns[stage-2], C_buf_next,pitch_c_buf,C_d,pitch_c, r_tex,jacobi_iterations,r_buf_next,pitch_r_buf,os);

					r_buf_curr=r_buf_next;
					C_buf_curr=C_buf_next;
				}
				else{
					jacobi_poisson_2D_device<F,F2,S>(jacobi_weight,alpha_curr,beta_curr,boundary_padding_thickness,n, C_buf_next,pitch_c_buf,C_d,pitch_c,B_tex,jacobi_iterations,B_d,pitch_b,os);

					/*Synchronize Buffer C_buf to contain the same values as C_d*/
					if ((jacobi_iterations%2) !=0){
						cudaError_t cpyerr=gpuErrorCheck(cudaMemcpy2D(C_d,pitch_c,C_buf,pitch_c_buf,n*sizeof(float2),n,cudaMemcpyDeviceToDevice),os);
						
						if (cpyerr!=cudaSuccess){
							return DesalStatus::CUDAError;
						}
						
					}
				}
			
			}
			
			//print_vector_field_k2<<<1,1>>>(n,n, C_d, pitch_c,'W');	
			return DesalStatus::Success;
			
		}

		template<class F, class F2, class S>
		DesalStatus mg_vc_poisson_2D_device(F alpha, F gamma, F eta, int boundary_padding_thickness, int n, F2* B_d, int pitch_b, F2* C_d, int pitch_c, S* os=nullptr){
			
			int jacobi_rounds=30;
			int multigrid_stages=10;//10 stages bei n=8000 geht net
			F jacobi_weight=1;
			
			F2* C_buf; //holding contents of intermediary U results of the various grid sizes
			F2* r_buf;

			size_t pitch_c_buf;
			size_t pitch_r_buf;
			
			double inv_power_of_two_lookup[]={1,1.0/2.0,1.0/4.0,1.0/8.0,1.0/16.0,1.0/32.0,1.0/64.0,1.0/128.0,1.0/256.0,1.0/512,1.0/1024,1.0/2048,1.0/4098,1.0/8196,1.0/16392,1.0/32784};
			
			int n_buf=ceil((2.0-0.9*inv_power_of_two_lookup[multigrid_stages-1])*n); //TODO: Calculate required stage more tightly. The 0.9 multiplication is just there to allocate a little bit more than required for robustness concerning numerical errors in the expression
			//int n_buf=2*n;
			printf("Result:%d\n",n_buf);
			
			cudaMallocPitch((void**)&C_buf,&pitch_c_buf,n*sizeof(F2),n_buf);
			cudaError_t err=gpuErrorCheck(cudaMemset2D(C_buf,pitch_c_buf,0.0,n*sizeof(F2),n_buf),os);
			
			if (err != cudaSuccess){
				return DesalStatus::CUDAError;
			}
			//cudaMemcpy2D(C_buf,pitch_c_buf,C_d,pitch_c,n*sizeof(F2),n,cudaMemcpyDeviceToDevice);
			
			cudaError_t err_malloc=gpuErrorCheck(cudaMallocPitch((void**)&r_buf,&pitch_r_buf,n*sizeof(F2),n_buf),os);
			
			
			if (err_malloc!= cudaSuccess){
				return DesalStatus::CUDAError;
			}
			
			DesalStatus res= mg_vc_poisson_2D_nobuf_device<F,F2>(alpha, gamma,eta, boundary_padding_thickness, n, B_d, pitch_b, C_d, pitch_c, C_buf, pitch_c_buf, r_buf, pitch_r_buf, jacobi_weight,jacobi_rounds,multigrid_stages, os);
		
			cudaFree(C_buf);
			cudaFree(r_buf);
			
			return res;
		
		}
	
	}

}