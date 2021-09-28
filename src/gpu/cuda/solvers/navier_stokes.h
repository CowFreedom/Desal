#pragma once
#include "../error_handling.h"

namespace desal{
	namespace cuda{
		
		template<class F, class F2, class S>
		DesalStatus navier_stokes_2D_nobuf_device(F dt, F dy, F dx, F m, F k, float*2 U_d, int pitch_u, float* Q_d, int pitch_q, //more buffers){
			//solve Advection
			advection_2D_f32_device(float dt, float dy, float dx,  int m_q,  int k_q, float2* U_d, int pitch_u, float* Q_d, int pitch_q, float* C_d, int pitch_c)
			//solve Diffusion
			
			
			//add forces
			return DesalStatus::Success;
		}
		
	}
}