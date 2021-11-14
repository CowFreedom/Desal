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
		
		namespace poisson_multigrid{
			namespace blocksizes{			
				namespace group1{
					constexpr int MX=512;
					constexpr int MY=2;		
				}
			}	
		}
	
		__global__
		void k_jacobi_poisson(float weight, float alpha, float beta_inv, int boundary_padding_thickness, int m, int k, cudaTextureObject_t X_old, float2* X_new, int pitch_x, cudaTextureObject_t B);
		__global__
		void print_vector_field_k2(int m,int k, float2* M, int pitch,char name, int iteration=0);

		//Solves AX=B
		template<class F, class F2, class S>
		__host__
		DesalStatus jacobi_poisson(F alpha, F beta, int boundary_padding_thickness, int m, int k, F2* X, int pitch_x, F2* X_buf, int pitch_x_buf, float2* B_d, int pitch_b,int max_jacobi_iterations, F jacobi_weight, F tol, F early_termination_tol, F* sos_residual_d, F* sos_residual_h, S* os=nullptr){
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
			texDesc.filterMode = cudaFilterModeLinear;
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
			
			//If error is already below threshold, we can quit here
			F sos_residual_h_prev;
			cudaError_t err2=reduce_sum_of_squares_poisson<F,F2>(alpha,beta,boundary_padding_thickness,m,k,X_tex, B_tex, sos_residual_d, 1);
			gpuErrorCheck(err2,os);	
	
			cudaMemcpy(&sos_residual_h_prev,sos_residual_d,sizeof(F),cudaMemcpyDeviceToHost);
			
			if (sos_residual_h_prev<=tol){
					*sos_residual_h=sos_residual_h_prev;
					return DesalStatus::Success;
			}		
			F beta_inv=1.0/beta;

			int threads_per_block_x=poisson_multigrid::blocksizes::group1::MX;	
			int threads_per_block_y=poisson_multigrid::blocksizes::group1::MY;
			int blocks_x=ceil(static_cast<float>(k)/(threads_per_block_x));
			int blocks_y=ceil(static_cast<float>(m)/(threads_per_block_y));

			dim3 threads=dim3(threads_per_block_x,threads_per_block_y,1);
			dim3 blocks=dim3(blocks_x,blocks_y,1);
			
			int iteration_blocks=max_jacobi_iterations/2;
			
			for (int i=0;i<iteration_blocks;i++){
			
				k_jacobi_poisson<<<blocks,threads>>>(jacobi_weight,alpha,beta_inv,boundary_padding_thickness,m,k,X_tex,X_buf,pitch_x_buf,B_tex);	
				reduce_sum_of_squares_poisson<F,F2>(alpha,beta,boundary_padding_thickness, m,k,X_buf_tex, B_tex, sos_residual_d, 1);					
				cudaMemcpy(sos_residual_h,sos_residual_d,sizeof(F)*1,cudaMemcpyDeviceToHost);		

				if (os){
					(*os)<<"Jacobi: Iteration: "<< 2*i+1<<" f_0: "<< *sos_residual_h <<"\n";
				}
				
				if ((*sos_residual_h)<=tol){
					return DesalStatus::Success;
				}
				else if (((*sos_residual_h)/sos_residual_h_prev)>early_termination_tol){
					if (os){
						(*os)<<"Jacobi: Iteration: "<< 2*i+1<<" terminated early due to diminising returns\n";
					}
					return DesalStatus::MathError;
				}
				k_jacobi_poisson<<<blocks,threads>>>(jacobi_weight,alpha,beta_inv,boundary_padding_thickness,m,k,X_buf_tex,X,pitch_x,B_tex);			
				sos_residual_h_prev=*sos_residual_h;
			}
			
			if (max_jacobi_iterations%2 ==0){
			
			}
			else{
			
				k_jacobi_poisson<<<blocks,threads>>>(jacobi_weight,alpha,beta_inv,boundary_padding_thickness,m,k,X_tex,X_buf,pitch_x_buf,B_tex);		
				err=gpuErrorCheck(cudaMemcpy2D(X,pitch_x,X_buf,pitch_x_buf,k*sizeof(F2),m,cudaMemcpyDeviceToDevice),os);
				
				if (err != cudaSuccess){
					return DesalStatus::CUDAError;
				}
				
			}
			reduce_sum_of_squares_poisson<F,F2>(alpha,beta,boundary_padding_thickness, m,k,X_tex, B_tex, sos_residual_d, 1);	
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
		DesalStatus mg_vc_poisson(F alpha, F gamma, F eta, int boundary_padding_thickness, int m,int k, F2* B_d, int pitch_b, F2* C_d, int pitch_c, F2** C_buf, size_t* pitch_c_buf, F2** r_buf, size_t* pitch_r_buf, int* max_jacobi_iterations, int multigrid_stages, F jacobi_weight, F tol , F early_termination_tol, F* sos_residual_d, F* sos_residual_h, S* os=nullptr){			
			F beta=gamma+eta;	
			
			if (os){				
					(*os)<<"V-Cycle: Stage 1\n";
			}
			
			DesalStatus jacobi_status=jacobi_poisson<F,F2,S>(alpha,beta,boundary_padding_thickness,m,k, C_d,pitch_c,C_buf[0],pitch_c_buf[0], B_d,pitch_b,max_jacobi_iterations[0],jacobi_weight,tol,early_termination_tol, sos_residual_d, sos_residual_h,os);
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

				//Solve new system U_{hr}x_{hr}=r_{h_r} on restricted grid hr=0.5h
				alpha_curr=0.25*alpha;
				gamma_curr=0.25*gamma;
				beta_curr=gamma_curr+eta;
				
				cudaError_t err=gpuErrorCheck(cudaMemcpy2D(C_d,pitch_c,C_buf_curr,pitch_c_buf[stage],k_buf[stage]*sizeof(float2),m_buf[stage],cudaMemcpyDeviceToDevice),os);
				if (err!=cudaSuccess){
					return DesalStatus::CUDAError;
				}

				jacobi_status=jacobi_poisson<F,F2,S>(alpha_curr,beta_curr,boundary_padding_thickness,m_buf[stage],k_buf[stage], C_buf_curr,pitch_c_buf[stage],C_d,pitch_c, r_buf_curr,pitch_r_buf[stage],max_jacobi_iterations[stage],jacobi_weight,tol,early_termination_tol,sos_residual_d, sos_residual_h,os);

				if (stage<(multigrid_stages-1)){
					transform_entries_into_residuals_device<F,F2>(alpha_curr,beta_curr, boundary_padding_thickness, m_buf[stage],k_buf[stage], C_buf_curr, pitch_c_buf[stage], r_buf_curr, pitch_r_buf[stage], r_buf[0], pitch_r_buf[0]); //TODO C_buf should be equal to C at this stage

					r_buf_prev=r_buf_curr;
					C_buf_curr=C_buf[stage+1];
					r_buf_curr=r_buf[stage+1];
					m_buf[stage+1]=restrict_n(m_buf[stage]);
					k_buf[stage+1]=restrict_n(k_buf[stage]);
				}
				
			}


			for (int stage=multigrid_stages;stage>1;stage--){
				if (os){				
					(*os)<<"V-Cycle Up: Stage "<< stage-1<<"\n";
				}
				alpha_curr=4*alpha;
				gamma_curr=4*gamma;
				beta_curr=gamma_curr+eta;
				F2* C_buf_next=C_buf[stage-2];

				//prolong previous restricted result and correct result at current gridsize				
				prolong_and_add<F,F2>(m_buf[stage-2], k_buf[stage-2], m_buf[stage-1], k_buf[stage-1], C_buf_next,pitch_c_buf[stage-2],C_buf_curr,pitch_c_buf[stage-1]);

				//relax again
				cudaError_t err=gpuErrorCheck(cudaMemcpy2D(C_d,pitch_c,C_buf_next,pitch_c_buf[stage-2],k_buf[stage-2]*sizeof(float2),m_buf[stage-2],cudaMemcpyDeviceToDevice),os);
				
				if (err!=cudaSuccess){
					return DesalStatus::CUDAError;
				}
				
				if (stage>2){
					r_buf_curr=r_buf[stage-2];
					jacobi_status=jacobi_poisson<F,F2,S>(alpha_curr,beta_curr,boundary_padding_thickness,m_buf[stage-2], k_buf[stage-2], C_buf_next,pitch_c_buf[stage-2],C_d,pitch_c,r_buf_curr,pitch_r_buf[stage-2],max_jacobi_iterations[stage-2],jacobi_weight,tol,early_termination_tol,sos_residual_d,sos_residual_h, os);				
					C_buf_curr=C_buf_next;
				}
				else{
					jacobi_status=jacobi_poisson<F,F2,S>(alpha_curr,beta_curr,boundary_padding_thickness,m,k,C_buf_next,pitch_c_buf[stage-2],C_d,pitch_c,B_d,pitch_b,max_jacobi_iterations[stage-2],jacobi_weight,tol,early_termination_tol,sos_residual_d,sos_residual_h,os);

					//Synchronize Buffer C_d to contain the same values as C_buf
					if ((max_jacobi_iterations[0]%2) !=0){
						cudaError_t cpyerr=gpuErrorCheck(cudaMemcpy2D(C_d,pitch_c,C_buf[0],pitch_c_buf[0],k*sizeof(float2),m,cudaMemcpyDeviceToDevice),os);
						
						if (cpyerr!=cudaSuccess){
							return DesalStatus::CUDAError;
						}
						
					}
				}
			}			
			return jacobi_status;
		}
		
		template<class F, class F2, class S>
		DesalStatus mg_vc_poisson(F alpha, F gamma, F eta, int boundary_padding_thickness, int m, int k, F2* B_d, size_t pitch_b, F2* C_d, size_t pitch_c, int* max_jacobi_iterations, int multigrid_stages, F jacobi_weight=1.0,F tol=0.1, F early_termination_tol=0.99, F* sos_residual=nullptr, S* os=nullptr){	
			float v=3*(1<<(multigrid_stages-1));
			
			if (multigrid_stages<1 || tol<0 || multigrid_stages >20 || m<v || k<v || early_termination_tol>1 || early_termination_tol<0){			
				if (os){
					if(m>(3*v) || k>(3*v)){
						(*os)<<"Too many multigrid stages for the input size\n";
					}
				}
				return DesalStatus::InvalidParameters;
			}
			
			int k_buf=k;
			int m_buf=m;			
	
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
			cudaMemcpy2D(C_buf[0],pitch_c_buf[0],C_d,pitch_c,k*sizeof(F2),m,cudaMemcpyDeviceToDevice); //Todo: Copy only boundary values
			
			F* sos_residual_d;
			F sos_residual_h=-1;
			
			//Allocates the buffer for the sum of squares reduction of the residual. Same threads_per_block as in reduce_sum_of_squares_poisson_residual
			constexpr int threads_per_block_x=reductions::blocksizes::group1::MX;
			constexpr int threads_per_block_y=reductions::blocksizes::group1::MY;
			int blocks_x=ceil(static_cast<float>(k)/(2.0*threads_per_block_x));
			int blocks_y=ceil(static_cast<float>(m)/(2.0*threads_per_block_y));
			size_t size_e=blocks_x*blocks_y*sizeof(F);
			cudaMalloc((void**) &sos_residual_d,size_e);
			cudaError_t err=gpuErrorCheck(cudaMemset(sos_residual_d,0,size_e),os);
				
			if (err != cudaSuccess){
				deallocate_buffer_array(C_buf,multigrid_stages);
				deallocate_buffer_array(r_buf,multigrid_stages);
				return DesalStatus::CUDAError;
			}

			DesalStatus res= mg_vc_poisson<F,F2,S>(alpha, gamma,eta, boundary_padding_thickness, m,k, B_d, pitch_b, C_d, pitch_c, C_buf, pitch_c_buf, r_buf, pitch_r_buf, max_jacobi_iterations,multigrid_stages, jacobi_weight,tol, early_termination_tol, sos_residual_d, &sos_residual_h, os);
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