#pragma once
#include "../error_handling.h"
#include "advection.h"
#include "poisson_multigrid.h"
#include "flow_by_forces.h"
#include "../reductions.h"
#include "../transformations.h"

namespace desal{
	namespace cuda{
		
		template<class F, class F2>
		inline DesalStatus navier_stokes_2D_nobuf_device(F dt, F viscousity, int boundary_padding_thickness, F dy, F dx,  int m_q,  int k_q, F2* U_d, int pitch_u, F* U_buf, int pitch_u_buf, F** MG_buf, int* pitch_mg_buf ,F** mg_r_buf, int* pitch_mg_r_buf, int* max_jacobi_iterations, int multigrid_stages, F jacobi_weight,  F mg_tol, F* mg_sos_error_d){//more buffers){
			//solve Advection
			advection_field_2D_device<F,F2>(dt,1,dy,dx,m,k,U_old,pitch_u_old,U_old,pitch_u_old,U_new,pitch_u_new);
			//solve Diffusion
			print_vector_field_k2<<<1,1>>>(m,k, U_old, pitch_u_old,'B');
			print_vector_field_k2<<<1,1>>>(m,k, U_new, pitch_u_new,'A');		
			float alpha=(dx*dy)/(viscousity*dt); //see manual for details
			float gamma=alpha; //see manual for details
			float eta=4.0; //see manual for details		
			
			
			DesalStatus mg_result=mg_vc_poisson_2D_nobuf_device<F,F2,S>(alpha, gamma, eta, boundary_padding_thickness, m,k, U_new, pitch_u_new, U_old, pitch_u_old, mg_U_buf, pitch_mg_u_buf, mg_r_buf, pitch_mg_r_buf, max_jacobi_iterations, multigrid_stages, jacobi_weight, mg_tol , mg_sos_residual_d, &mg_sos_residual_h, os);	

			return mg_result;
			
		}
		
		
		template<class F, class F2, class S=void>
		DesalStatus navier_stokes_2D_device(F dt, F viscousity, int boundary_padding_thickness, F dy, F dx,  int m,  int k, F2* U_old, int pitch_u_old, F2* F_d, size_t pitch_f, F2* U_new, size_t pitch_u_new, int* max_jacobi_iterations, int multigrid_stages, F jacobi_weight,  F mg_tol, S* os=nullptr){
			float v=3*(1<<(multigrid_stages-1));
			
			if (multigrid_stages<1 || mg_tol<0 || multigrid_stages >20 || m<v || k<v){
				return DesalStatus::InvalidParameters;
			}
			
			int k_buf=k;
			int m_buf=m;
			
			//Allocate Buffers for the multigrid Method
			F2* mg_U_buf[20];
			F2* mg_r_buf[20];
			size_t pitch_mg_u_buf[20];
			size_t pitch_mg_r_buf[20];
			
			cudaError_t err1,err2,err3,err4;
			
				
				
			for (int i=0; i<multigrid_stages;i++){
				err1=cudaMallocPitch((void**)(&mg_U_buf[i]),&pitch_mg_u_buf[i],k_buf*sizeof(F2),m_buf);
				
				err2=gpuErrorCheck(cudaMallocPitch((void**)&mg_r_buf[i],&pitch_mg_r_buf[i],k_buf*sizeof(F2),m_buf),os);

				err3=gpuErrorCheck(cudaMemset2D(mg_U_buf[i],pitch_mg_u_buf[i],0.0,k_buf*sizeof(F2),m_buf),os);
				err4=gpuErrorCheck(cudaMemset2D(mg_r_buf[i],pitch_mg_r_buf[i],0.0,k_buf*sizeof(F2),m_buf),os);
				
				if ((err1 != cudaSuccess)&&(err2 != cudaSuccess)&&(err3 != cudaSuccess)&&(err4 != cudaSuccess)){
					deallocate_buffer_array(mg_U_buf,i);
					deallocate_buffer_array(mg_r_buf,i);
					return DesalStatus::CUDAError;
				}
				
				m_buf=restrict_n(m_buf);
				k_buf=restrict_n(k_buf);
			}
			F mg_sos_residual_h=0;
			
			F* mg_sos_residual_d;
			
			constexpr int threads_per_block_x=8;
			constexpr int threads_per_block_y=4;
			int blocks_x=ceil(static_cast<F>(k)/(2*threads_per_block_x));
			int blocks_y=ceil(static_cast<F>(m)/(2*threads_per_block_y));
			size_t size_e=blocks_x*blocks_y*sizeof(F);
			cudaMalloc((void**) &mg_sos_residual_d,size_e);
			cudaError_t err=gpuErrorCheck(cudaMemset(mg_sos_residual_d,0,size_e),os);
				
			if (err != cudaSuccess){
				deallocate_buffer_array(mg_U_buf,multigrid_stages);
				deallocate_buffer_array(mg_r_buf,multigrid_stages);
				return DesalStatus::CUDAError;
			}

			advection_field_2D_device<F,F2>(dt,1,dy,dx,m,k,U_old,pitch_u_old,U_old,pitch_u_old,U_new,pitch_u_new);
			//solve Diffusion
		//	print_vector_field_k2<<<1,1>>>(m,k, U_old, pitch_u_old,'B');
		//	print_vector_field_k2<<<1,1>>>(m,k, U_new, pitch_u_new,'A');		
			float alpha=(dx*dy)/(viscousity*dt); //see manual for details
			float gamma=alpha; //see manual for details
			float eta=4.0; //see manual for details		
			
			
			DesalStatus mg_result=mg_vc_poisson_2D_nobuf_device<F,F2,S>(alpha, gamma, eta, boundary_padding_thickness, m,k, U_new, pitch_u_new, U_old, pitch_u_old, mg_U_buf, pitch_mg_u_buf, mg_r_buf, pitch_mg_r_buf, max_jacobi_iterations, multigrid_stages, jacobi_weight, mg_tol , mg_sos_residual_d, &mg_sos_residual_h, os);	

			add_forces(dt, boundary_padding_thickness, m, k,U_old,pitch_u_old, F_d, pitch_f, U_new, pitch_u_new);
	

			alpha=dx*dy; //see manual for details
			gamma=0; //see manual for details
			eta=4.0; //see manual for details	

			F2* P;
			size_t pitch_p;
			err1=cudaMallocPitch((void**)(&P),&pitch_p,k*sizeof(F2),m);
			cudaMemset2D(P,pitch_p,0.0,sizeof(float2)*k,m);				
			//solve laplace equation for pressure
			
			divergence(dy, dx, boundary_padding_thickness,m,k,U_new, pitch_u_new, U_old, pitch_u_old);
		
			
			mg_result=mg_vc_poisson_2D_nobuf_device<F,F2,S>(alpha, gamma, eta, boundary_padding_thickness, m,k,U_old, pitch_u_old, P, pitch_p, mg_U_buf, pitch_mg_u_buf, mg_r_buf, pitch_mg_r_buf, max_jacobi_iterations, multigrid_stages, jacobi_weight, mg_tol , mg_sos_residual_d, &mg_sos_residual_h, os);	

			subtract_gradient(dy, dx, boundary_padding_thickness,m,k,P, pitch_p, U_new, pitch_u_new);
			
			//DesalStatus mg_result=DesalStatus::Success;
			deallocate_buffer_array(mg_U_buf,multigrid_stages);
			deallocate_buffer_array(mg_r_buf,multigrid_stages);
			cudaFree(P);
			cudaFree(mg_sos_residual_d);		
			
			
			
			return mg_result;
			
		}
		
		
	}
}