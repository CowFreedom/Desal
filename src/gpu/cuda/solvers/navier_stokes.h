#pragma once
#include "../error_handling.h"
#include "advection.h"
#include "poisson_multigrid.h"

namespace desal{
	namespace cuda{
		
		template<class F, class F2>
		DesalStatus navier_stokes_2D_nobuf_device(F dt, int boundary_padding_thickness, F dy, F dx,  int m_q,  int k_q, F2* U_d, int pitch_u, F* U_buf, int pitch_u_buf, F** MG_buf, int pitch_mg_buf ,F* r_buf, int pitch_r_buf, int* max_jacobi_iterations, int multigrid_stages, F jacobi_weight,  F mg_tol, F* mg_sos_error_d){//more buffers){
			//solve Advection
			advection_2D_f32_device(dt,poundary_padding_thickness,dy,dx,m_q,k_q,U_d,pitch_u,U_d,pitch_u,U_buf,pitch_u_buf);
			//solve Diffusion
			
			F mg_sos_error_h=0;
			
			mg_vc_poisson_2D_nobuf_device(alpha, gamma, eta, boundary_padding_thickness, m_q*k_q, U_d, pitch_u, U_buf, pitch_u_buf, MG_buf, pitch_mg_buf, r_buf, pitch_r_buf, max_jacobi_iterations, multigrid_stages, jacobi_weight, mg_tol , mg_sos_error_d, mg_sos_error_h);	
//add forces
			return DesalStatus::Success;
		}
		
	}
}