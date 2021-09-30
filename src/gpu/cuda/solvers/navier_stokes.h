#pragma once
#include "../error_handling.h"
//#include "advection.h"
//#include "poisson_multigrid.h"
#include "../reductions.h"
#include "../transformations.h"

namespace desal{
	namespace cuda{
		
		template<class F, class F2>
		DesalStatus navier_stokes_2D_nobuf_device(F dt, F viscousity, int boundary_padding_thickness, F dy, F dx,  int m_q,  int k_q, F2* U_d, int pitch_u, F* U_buf, int pitch_u_buf, F** MG_buf, int* pitch_mg_buf ,F** mg_r_buf, int* pitch_mg_r_buf, int* max_jacobi_iterations, int multigrid_stages, F jacobi_weight,  F mg_tol, F* mg_sos_error_d){//more buffers){
			//solve Advection
		//	advection_2D_f32_device(dt,poundary_padding_thickness,dy,dx,m_q,k_q,U_d,pitch_u,U_d,pitch_u,U_buf,pitch_u_buf);
			//solve Diffusion
			
			float alpha=(dx*dy)/(viscousity*dt); //see manual for details
			float gamma=alpha; //see manual for details
			float eta=4.0; //see manual for details			
			F mg_sos_error_h=0;
			
			//DesalStatus mg_result=mg_vc_poisson_2D_nobuf_device(alpha, gamma, eta, boundary_padding_thickness, m_q*k_q, U_d, pitch_u, U_buf, pitch_u_buf, mg_U_buf, pitch_mg_buf, mg_r_buf, pitch_mg_r_buf, max_jacobi_iterations, multigrid_stages, jacobi_weight, mg_tol , mg_sos_error_d, mg_sos_error_h);	
			
			return mg_result;
			
		}
		
		
		template<class F, class F2, class S=void>
		DesalStatus navier_stokes_2D_device(F dt, F viscousity, int boundary_padding_thickness, F dy, F dx,  int m,  int k, F2* U_old, int pitch_u_old, F2* U_new, int pitch_u_new, int* max_jacobi_iterations, int multigrid_stages, F jacobi_weight,  F mg_tol, S* os=nullptr){//more buffers){
			float v=3*(1<<(multigrid_stages-1));
			
			if (multigrid_stages<1 || mg_tol<0 || multigrid_stages >20 || m<v || k<v){
				return DesalStatus::InvalidParameters;
			}
			
			int k_buf=k;
			int m_buf=m;
			
			//Allocate Buffers for the multigrid Method
			F2* mg_U_buf[20];
			F2* mg_r_buf[20];
			
			size_t pitch_mg_U_buf[20];
			size_t pitch_mg_r_buf[20];
			
			cudaError_t err1,err2,err3,err4;
				
			for (int i=0; i<multigrid_stages;i++){
				err1=cudaMallocPitch((void**)(&mg_U_buf[i]),&pitch_mg_U_buf[i],k_buf*sizeof(F2),m_buf);
				
				err2=gpuErrorCheck(cudaMallocPitch((void**)&mg_r_buf[i],&pitch_mg_r_buf[i],k_buf*sizeof(F2),m_buf),os);

				err3=gpuErrorCheck(cudaMemset2D(mg_U_buf[i],pitch_mg_U_buf[i],0.0,k_buf*sizeof(F2),m_buf),os);
				err4=gpuErrorCheck(cudaMemset2D(mg_r_buf[i],pitch_mg_r_buf[i],0.0,k_buf*sizeof(F2),m_buf),os);
				
				if ((err1 != cudaSuccess)&&(err2 != cudaSuccess)&&(err3 != cudaSuccess)&&(err4 != cudaSuccess)){
					deallocate_buffer_array(mg_U_buf,i);
					deallocate_buffer_array(mg_r_buf,i);
					return DesalStatus::CUDAError;
				}
				
				m_buf=restrict_n(m_buf);
				k_buf=restrict_n(k_buf);
			}
			

			//solve Advection
			//advection_2D_f32_device(dt,poundary_padding_thickness,dy,dx,m_q,k_q,U_d,pitch_u,U_d,pitch_u,U_buf,pitch_u_buf);
			//solve Diffusion
			
			float alpha=(dx*dy)/(viscousity*dt); //see manual for details
			float gamma=alpha; //see manual for details
			float eta=4.0; //see manual for details			
			F mg_sos_error_h=0;
			
			//DesalStatus mg_result=mg_vc_poisson_2D_nobuf_device(alpha, gamma, eta, boundary_padding_thickness, m_q*k_q, U_d, pitch_u, U_buf, pitch_u_buf, MG_buf, pitch_mg_buf, mg_r_buf, pitch_mg_r_buf, max_jacobi_iterations, multigrid_stages, jacobi_weight, mg_tol , mg_sos_error_d, mg_sos_error_h);	
			DesalStatus mg_result=DesalStatus::Success;
			
			
			
			return mg_result;
			
		}
		
		
	}
}