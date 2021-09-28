#pragma once

namespace desal{
	namespace cuda{
		
		template<class F, class F2>
		__host__
		cudaError_t reduce_sum_of_squares_poisson_field_residual_device(F alpha, F beta, int boundary_padding_thickness, int n, F2* A_d,int pitch_a, cudaTextureObject_t B_tex, F* r_d, int stride_r);

		template<class F, class F2>
		__host__
		cudaError_t reduce_sum_of_squares_poisson_field_residual_device(F alpha, F beta, int boundary_padding_thickness, int n, F2* A_d,int pitch_a, F2* B_d, int pitch_b, F* r_d, int stride_r);

		template<class F>
		__host__
		void reduce_sum_device(int n, F* r_d, int stride_r);	
		//AX=B
	
		//void reduce_sum_of_squares_poisson_field_residual_device(F alpha, F beta, int boundary_padding_thickness, int n, F2* A_d,int pitch_a, cudaTextureObject_t B_tex, F* r_d, int stride_r);
/*
		void reduce_sum_f32_device(int n, float* r_d, int stride_r);

		void reduce_sum_f64_device(int n, double* r_d, int stride_r);
*/
		inline int restrict_n(int n){

			if ((n%2) ==false){
				return (n-1)*0.5+1;
			}
			else{ 
				return 0.5*(n-2)+2;
			}
			
		}

		template<class F, class F2>
		cudaError_t restrict(int n, int n_res, F2* dest, int pitch_dest, F2* src, int pitch_src);
	}
}