#pragma once

namespace desal{
	namespace cuda{
		
		
		template<class F, class F2>
		DesalStatus advection_field_2D_device(F dt, int boundary_padding_thickness, F dy, F dx,  int m_q,  int k_q, F2* U_d, int pitch_u, F2* Q_d, int pitch_q, F2* C_d, int pitch_c);
	}
}