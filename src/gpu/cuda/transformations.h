#pragma once

namespace desal{
	
	namespace cuda{

		
		template<class F, class F2>
		__host__
		cudaError_t transform_entries_into_residuals_device(F alpha, F beta, int boundary_padding_thickness, int m, int k, F2* A_d, int pitch_a, F2* B_d, int pitch_b, F2* r_d, int pitch_r);


		__host__
		cudaError_t transform_entries_into_residuals_device(float alpha,float beta, int boundary_padding_thickness, int m, int k, float2* A_d, int pitch_a, float2* B_d, int pitch_b, float2* r_d, int pitch_r);


		//template<class F2>
		__host__
		cudaError_t prolong_and_add(int n_p, int n_r, float2* dest, int pitch_dest, float2* src, int pitch_s);
	}
}