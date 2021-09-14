//AX=B

#include "transformations.h"


__host__
void transform_entries_into_square_residuals_f32_device(float alpha, float beta, int boundary_padding_thickness, int m, int k, float2* A_d, int pitch_a, float2* B_d, int pitch_b, float2* r_d, int pitch_r){

	transform_entries_into_square_residuals_device<float,float2>(alpha, beta, boundary_padding_thickness, m, k, A_d, pitch_a, B_d, pitch_b, r_d, pitch_r);

}