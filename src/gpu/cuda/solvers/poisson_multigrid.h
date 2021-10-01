#pragma once
#include "../error_handling.h"
#include "../reductions.h"
#include "../transformations.h"
#include <iostream>
/*
template<class S>
DesalStatus mg_vc_poisson_2D_f32_device(float alpha, float gamma, float eta, int boundary_padding_thickness, int n, float2* B_d, int pitch_b, float2* C_d, int pitch_c, S* os=nullptr);
*/

namespace desal {
	
	namespace cuda{
	
		template<class F, class F2>
		__global__
		void k_jacobi_poisson_2D(F weight, F alpha, F beta_inv, int boundary_padding_thickness, int m, int k, cudaTextureObject_t X_old, F2* X_new, int pitch_x, cudaTextureObject_t B);

		__global__
		void print_vector_field_k2(int m,int k, float2* M, int pitch,char name, int iteration=0);

		//Solves AX=B
		template<class F, class F2, class S>
		__host__
		DesalStatus jacobi_poisson_2D_device(F alpha, F beta, int boundary_padding_thickness, int m, int k, F2* X, int pitch_x, F2* X_buf, int pitch_x_buf, float2* B_d, int pitch_b,int max_jacobi_iterations, F jacobi_weight, F tol, F* sos_residual_d, F* sos_residual_h, S* os=nullptr){

			//If error is already below threshold, we can quit immediately
			F sos_residual_h_prev;
			reduce_sum_of_squares_poisson_field_residual_device<F,F2>(alpha,beta,boundary_padding_thickness, m,k,X,pitch_x, B_d, pitch_b, sos_residual_d, 1);	
//print_vector_field_k2<<<1,1>>>(n,n,X,pitch_x,'O');				
			cudaMemcpy(&sos_residual_h_prev,sos_residual_d,sizeof(F)*1,cudaMemcpyDeviceToHost);

			if (sos_residual_h_prev<=tol){
					*sos_residual_h=sos_residual_h_prev;
					return DesalStatus::Success;
			}		
			F beta_inv=1.0/beta;
			
			//print_vector_field_k2<<<1,1>>>(n,n, X, pitch_x,'L');
			//Create Resource description

			//Create Resource description
			cudaResourceDesc resDescB;
			memset(&resDescB,0,sizeof(resDescB));
			resDescB.resType = cudaResourceTypePitch2D;
			resDescB.res.pitch2D.devPtr=B_d;
			resDescB.res.pitch2D.width=k;
			resDescB.res.pitch2D.height=m;
			resDescB.res.pitch2D.pitchInBytes=pitch_b;
			resDescB.res.pitch2D.desc=cudaCreateChannelDesc<F2>(); 
		
			cudaResourceDesc resDescX_buf;
			memset(&resDescX_buf,0,sizeof(resDescX_buf));
			resDescX_buf.resType = cudaResourceTypePitch2D;
			resDescX_buf.res.pitch2D.devPtr=X_buf;
			resDescX_buf.res.pitch2D.pitchInBytes=pitch_x_buf;
			resDescX_buf.res.pitch2D.width=k;
			resDescX_buf.res.pitch2D.height=m;
			resDescX_buf.res.pitch2D.desc=cudaCreateChannelDesc<F2>(); //cudaCreateChannelDesc(32,32,0,0,cudaChannelFormatKindFloat) is equivalent cudaCreateChannelDesc<float2>()

			cudaResourceDesc resDescX;
			memset(&resDescX,0,sizeof(resDescX));
			resDescX.resType = cudaResourceTypePitch2D;
			resDescX.res.pitch2D.devPtr=X;
			resDescX.res.pitch2D.pitchInBytes=pitch_x;
			resDescX.res.pitch2D.width=k;
			resDescX.res.pitch2D.height=m;
			resDescX.res.pitch2D.desc=cudaCreateChannelDesc<F2>(); //cudaCreateChannelDesc(32,32,0,0,cudaChannelFormatKindFloat) is equivalent cudaCreateChannelDesc<float2>()


			//Create Texture description
			cudaTextureDesc texDesc;
			memset(&texDesc,0,sizeof(texDesc));
			texDesc.normalizedCoords = false;
			texDesc.filterMode = cudaFilterModeLinear; //change to nearest
			texDesc.readMode=cudaReadModeElementType;
			texDesc.addressMode[0] = cudaAddressModeBorder;
			texDesc.addressMode[1] = cudaAddressModeBorder;

			cudaTextureObject_t B_tex;			
			cudaTextureObject_t X_buf_tex;
			
			cudaTextureObject_t X_tex;
			
			cudaError_t err=gpuErrorCheck(cudaCreateTextureObject(&B_tex, &resDescB, &texDesc, NULL),os);
			if (err !=cudaSuccess){
				return DesalStatus::CUDAError;
			}
			
			err=cudaCreateTextureObject(&X_buf_tex, &resDescX_buf, &texDesc, NULL);
			if (err != cudaSuccess){
					return DesalStatus::CUDAError;
			}
				
			err=cudaCreateTextureObject(&X_tex, &resDescX, &texDesc, NULL);
			if (err != cudaSuccess){
					return DesalStatus::CUDAError;
			}
		//print_vector_field_k2<<<1,1>>>(n,n,B_d,pitch_b,'B');

			int threads_per_block_x=512;	
			int threads_per_block_y=2;	
			int blocks_x=ceil(static_cast<float>(k)/(threads_per_block_x));
			int blocks_y=ceil(static_cast<float>(m)/(threads_per_block_y));
			F early_termination_tol=0.99;
			dim3 threads=dim3(threads_per_block_x,threads_per_block_y,1);
			dim3 blocks=dim3(blocks_x,blocks_y,1);
			int iteration_blocks=max_jacobi_iterations/2;
			//print_vector_field_k2<<<1,1>>>(n,n, X, pitch_x,'L');	
				
			for (int i=0;i<iteration_blocks;i++){
			
				k_jacobi_poisson_2D<F,F2><<<blocks,threads>>>(jacobi_weight,alpha,beta_inv,boundary_padding_thickness,m,k,X_tex,X_buf,pitch_x_buf,B_tex);	
//
				reduce_sum_of_squares_poisson_field_residual_device<F,F2>(alpha,beta,boundary_padding_thickness, m,k,X_buf,pitch_x_buf, B_d, pitch_b, sos_residual_d, 1);	
				
				cudaMemcpy(sos_residual_h,sos_residual_d,sizeof(F)*1,cudaMemcpyDeviceToHost);		
				if (os){
					(*os)<<"Jacobi: Iteration: "<< 2*i+1<<" f_0: "<< *sos_residual_h <<"\n";
				}
				
				//print_vector_field_k2<<<1,1>>>(n,n,X_buf,pitch_x_buf,'O');
				
				if ((*sos_residual_h)<=tol){
					return DesalStatus::Success;
				}
				else if (((*sos_residual_h)/sos_residual_h_prev)>early_termination_tol){
					if (os){
						(*os)<<"Jacobi: Iteration: "<< 2*i+1<<" terminated early due to diminising returns\n";
					}
					return DesalStatus::MathError;
				}
				k_jacobi_poisson_2D<F,F2><<<blocks,threads>>>(jacobi_weight,alpha,beta_inv,boundary_padding_thickness,m,k,X_buf_tex,X,pitch_x,B_tex);			
				sos_residual_h_prev=*sos_residual_h;
			}
			//TODO: No need to always keep both buffers synchronized
			if (max_jacobi_iterations%2 ==0){
			
			}
			else{
			
				k_jacobi_poisson_2D<F,F2><<<blocks,threads>>>(jacobi_weight,alpha,beta_inv,boundary_padding_thickness,m,k,X_tex,X_buf,pitch_x_buf,B_tex);		
				err=gpuErrorCheck(cudaMemcpy2D(X,pitch_x,X_buf,pitch_x_buf,k*sizeof(F2),m,cudaMemcpyDeviceToDevice),os);
				
				if (err != cudaSuccess){
					return DesalStatus::CUDAError;
				}
				
			}
			reduce_sum_of_squares_poisson_field_residual_device<F,F2>(alpha,beta,boundary_padding_thickness, m,k,X,pitch_x, B_d, pitch_b, sos_residual_d, 1);	
			cudaMemcpy(sos_residual_h,sos_residual_d,sizeof(F)*1,cudaMemcpyDeviceToHost);
			
			if (os){				
					(*os)<<"Jacobi: Iteration: "<< max_jacobi_iterations<<" f_0: "<< *sos_residual_h <<"\n";
			}
			
			if ((*sos_residual_h)<=tol){
				return DesalStatus::Success;
			}
			else{
				return DesalStatus::MathError;
			}
		}
		
		//AC=B
		template<class F, class F2, class S>
		__host__
		DesalStatus mg_vc_poisson_2D_nobuf_device(F alpha, F gamma, F eta, int boundary_padding_thickness, int m,int k, F2* B_d, int pitch_b, F2* C_d, int pitch_c, F2** C_buf, size_t* pitch_c_buf, F2** r_buf, size_t* pitch_r_buf, int* max_jacobi_iterations, int multigrid_stages, F jacobi_weight, F tol , F* sos_residual_d, F* sos_residual_h, S* os=nullptr){			
			F beta=gamma+eta;	
			
			if (os){				
					(*os)<<"V-Cycle: Stage 1\n";
			}
//print_vector_field_k2<<<1,1>>>(n,n,C_d,pitch_c,'O',2);			
			DesalStatus jacobi_status=jacobi_poisson_2D_device<F,F2,S>(alpha,beta,boundary_padding_thickness,m,k, C_d,pitch_c,C_buf[0],pitch_c_buf[0], B_d,pitch_b,max_jacobi_iterations[0],jacobi_weight,tol, sos_residual_d, sos_residual_h,os);
			if (jacobi_status==DesalStatus::Success){
				return jacobi_status;
			}
			transform_entries_into_residuals_device<F,F2>(alpha,beta, boundary_padding_thickness, m,k, C_buf[0], pitch_c_buf[0], B_d, pitch_b, r_buf[0], pitch_r_buf[0]); //TODO C_buf should be equal to C at this stage
	
			int m_buf[20];
			int k_buf[20];
			m_buf[0]=m;
			m_buf[1]=restrict_n(m);
			k_buf[0]=k;
			k_buf[1]=restrict_n(k);
		
			F2* r_buf_prev=r_buf[0];
			F alpha_curr=alpha;
			F beta_curr;
			F gamma_curr=gamma;
			F2* C_buf_curr=C_buf[1];	
			F2* r_buf_curr=r_buf[1];
		//	printf("Down Cycle\n");
		
			/*Synchronize Buffer C_buf to contain the same values as C_d*/
			if ((max_jacobi_iterations[0]%2) == 0){
				cudaError_t err=gpuErrorCheck(cudaMemcpy2D(C_buf[0],pitch_c_buf[0],C_d,pitch_c,k*sizeof(float2),m,cudaMemcpyDeviceToDevice),os);
				if (err != cudaSuccess){
					return DesalStatus::CUDAError;
				}
			}	
		
			for (int stage=1;stage<multigrid_stages;stage++){
				if (os){				
					(*os)<<"V-Cycle Down: Stage "<< stage+1<<"\n";
				}	
				//Restrict previous residuals
				restrict<F,F2>(m_buf[stage-1],k_buf[stage-1], m_buf[stage], k_buf[stage], r_buf_curr, pitch_r_buf[stage], r_buf_prev, pitch_r_buf[stage-1]);

				//print_vector_field_k2<<<1,1>>>(ns[stage],ns[stage], r_buf[stage], pitch_r_buf[stage],'L');
				//cudaDeviceSynchronize();
				//Solve new system Ax=r
		
				alpha_curr=0.25*alpha;
				gamma_curr=0.25*gamma;
				beta_curr=gamma_curr+eta;
				
				cudaError_t err=gpuErrorCheck(cudaMemcpy2D(C_d,pitch_c,C_buf_curr,pitch_c_buf[stage],k_buf[stage]*sizeof(float2),m_buf[stage],cudaMemcpyDeviceToDevice),os);
				if (err!=cudaSuccess){
					return DesalStatus::CUDAError;
				}

				jacobi_status=jacobi_poisson_2D_device<F,F2,S>(alpha_curr,beta_curr,boundary_padding_thickness,m_buf[stage],k_buf[stage], C_buf_curr,pitch_c_buf[stage],C_d,pitch_c, r_buf_curr,pitch_r_buf[stage],max_jacobi_iterations[stage],jacobi_weight,tol,sos_residual_d, sos_residual_h,os);

				if (stage<(multigrid_stages-1)){
					transform_entries_into_residuals_device<F,F2>(alpha_curr,beta_curr, boundary_padding_thickness, m_buf[stage],k_buf[stage], C_buf_curr, pitch_c_buf[stage], r_buf_curr, pitch_r_buf[stage], r_buf[0], pitch_r_buf[0]); //TODO C_buf should be equal to C at this stage

					//Iterate pointers
					r_buf_prev=r_buf_curr;
					C_buf_curr=C_buf[stage+1];
					r_buf_curr=r_buf[stage+1];
					m_buf[stage+1]=restrict_n(m_buf[stage]);//(ns-1)*0.5+1; //there are ns-1 spaces between nodes
					k_buf[stage+1]=restrict_n(k_buf[stage]);//(ns-1)*0.5+1; //there are ns-1 spaces between nodes
				}
				
			}

		
			//printf("Up Cycle\n");
			for (int stage=multigrid_stages;stage>1;stage--){
				if (os){				
					(*os)<<"V-Cycle Up: Stage "<< stage-1<<"\n";
				}
				alpha_curr=4*alpha;
				gamma_curr=4*gamma;
				beta_curr=gamma_curr+eta;
				F2* C_buf_next=C_buf[stage-2];

				//prolongate and correct previous result
				
				prolong_and_add(m_buf[stage-2], k_buf[stage-2], m_buf[stage-1], k_buf[stage-1], C_buf_next,pitch_c_buf[stage-2],C_buf_curr,pitch_c_buf[stage-1]);

				//k_check_boundary<<<1,1>>>(ns[stage-2],r_buf_next, pitch_r_buf[stage-2],0.0);
				//relax again
				cudaError_t err=gpuErrorCheck(cudaMemcpy2D(C_d,pitch_c,C_buf_next,pitch_c_buf[stage-2],k_buf[stage-2]*sizeof(float2),m_buf[stage-2],cudaMemcpyDeviceToDevice),os);
				
				if (err!=cudaSuccess){
					return DesalStatus::CUDAError;
				}
				
				if (stage>2){
					r_buf_curr=r_buf[stage-2];
					jacobi_status=jacobi_poisson_2D_device<F,F2,S>(alpha_curr,beta_curr,boundary_padding_thickness,m_buf[stage-2], k_buf[stage-2], C_buf_next,pitch_c_buf[stage-2],C_d,pitch_c,r_buf_curr,pitch_r_buf[stage-2],max_jacobi_iterations[stage-2],jacobi_weight,tol,sos_residual_d,sos_residual_h, os);				
					C_buf_curr=C_buf_next;
				}
				else{
					jacobi_status=jacobi_poisson_2D_device<F,F2,S>(alpha_curr,beta_curr,boundary_padding_thickness,m,k,C_buf_next,pitch_c_buf[stage-2],C_d,pitch_c,B_d,pitch_b,max_jacobi_iterations[stage-2],jacobi_weight,tol,sos_residual_d,sos_residual_h,os);

					//Synchronize Buffer C_d to contain the same values as C_buf
					if ((max_jacobi_iterations[0]%2) !=0){
						cudaError_t cpyerr=gpuErrorCheck(cudaMemcpy2D(C_d,pitch_c,C_buf[0],pitch_c_buf[0],k*sizeof(float2),m,cudaMemcpyDeviceToDevice),os);
						
						if (cpyerr!=cudaSuccess){
							return DesalStatus::CUDAError;
						}
						
					}
				}
			
			}
			
			
			//print_vector_field_k2<<<1,1>>>(m,k, C_d, pitch_c,'W');	
			return jacobi_status;
		}
		
		template<class F, class F2, class S>
		DesalStatus mg_vc_poisson_2D_device(F alpha, F gamma, F eta, int boundary_padding_thickness, int m, int k, F2* B_d, size_t pitch_b, F2* C_d, size_t pitch_c, int* max_jacobi_iterations, int multigrid_stages, F jacobi_weight=1.0,F tol=0.1, F* sos_residual=nullptr, S* os=nullptr){
		
			float v=3*(1<<(multigrid_stages-1));
			
			if (multigrid_stages<1 || tol<0 || multigrid_stages >20 || m<v || k<v){			
				if(m>(3*v) || k>(3*v)){
					(*os)<<"Too many multigrid stages for the input size\n";
				}
				return DesalStatus::InvalidParameters;
			}
			
			int k_buf=k;
			int m_buf=m;			
	
			//double inv_power_of_two_lookup[]={1,1.0/2.0,1.0/4.0,1.0/8.0,1.0/16.0,1.0/32.0,1.0/64.0,1.0/128.0,1.0/256.0,1.0/512,1.0/1024,1.0/2048,1.0/4098,1.0/8196,1.0/16392,1.0/32784};
			
			//int n_buf=ceil((2.0-0.9*inv_power_of_two_lookup[multigrid_stages-1])*n); //TODO: Calculate required stage more tightly. The 0.9 multiplication is just there to allocate a little bit more than required for robustness concerning numerical errors in the expression
			//int n_buf=2*n;
		//	printf("Result:%d\n",n_buf);
		
			F2* C_buf[20];
			F2* r_buf[20];
			size_t pitch_c_buf[20];
			size_t pitch_r_buf[20];
			
			cudaError_t err1,err2,err3,err4;	
			
			for (int i=0; i<multigrid_stages;i++){
				err1=cudaMallocPitch((void**)(&C_buf[i]),&pitch_c_buf[i],k_buf*sizeof(F2),m_buf);
				
				err2=gpuErrorCheck(cudaMallocPitch((void**)&r_buf[i],&pitch_r_buf[i],k_buf*sizeof(F2),m_buf),os);

				err3=gpuErrorCheck(cudaMemset2D(C_buf[i],pitch_c_buf[i],0.0,k_buf*sizeof(F2),m_buf),os);
				err4=gpuErrorCheck(cudaMemset2D(r_buf[i],pitch_r_buf[i],0.0,k_buf*sizeof(F2),m_buf),os);
				
				if ((err1 != cudaSuccess)&&(err2 != cudaSuccess)&&(err3 != cudaSuccess)&&(err4 != cudaSuccess)){
					deallocate_buffer_array(C_buf,i);
					deallocate_buffer_array(r_buf,i);
					return DesalStatus::CUDAError;
				}
				
				m_buf=restrict_n(m_buf);
				k_buf=restrict_n(k_buf);
			}		
			

			F* sos_residual_d;
			F sos_residual_h=-1;
			//Same threads_per_block as reduce_sum_of_squares_poisson_field_residual_device
			constexpr int threads_per_block_x=8;
			constexpr int threads_per_block_y=4;
			int blocks_x=ceil(static_cast<F>(k)/(2*threads_per_block_x));
			int blocks_y=ceil(static_cast<F>(m)/(2*threads_per_block_y));
			size_t size_e=blocks_x*blocks_y*sizeof(F);
			cudaMalloc((void**) &sos_residual_d,size_e);
			cudaError_t err=gpuErrorCheck(cudaMemset(sos_residual_d,0,size_e),os);
				
			if (err != cudaSuccess){
				deallocate_buffer_array(C_buf,multigrid_stages);
				deallocate_buffer_array(r_buf,multigrid_stages);
				return DesalStatus::CUDAError;
			}


			DesalStatus res= mg_vc_poisson_2D_nobuf_device<F,F2,S>(alpha, gamma,eta, boundary_padding_thickness, m,k, B_d, pitch_b, C_d, pitch_c, C_buf, pitch_c_buf, r_buf, pitch_r_buf, max_jacobi_iterations,multigrid_stages, jacobi_weight,tol,sos_residual_d, &sos_residual_h, os);
			//DesalStatus res=DesalStatus::Success;
			deallocate_buffer_array(C_buf,multigrid_stages);
			deallocate_buffer_array(r_buf,multigrid_stages);
			
			if (sos_residual){
				*sos_residual=sos_residual_h;
			}
			cudaFree(sos_residual_d);
			return res;
		
		
		}
	
	}

}