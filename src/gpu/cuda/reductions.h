#pragma once
//AX=B
void reduce_sum_of_squares_poisson_field_residual_f32_device(float alpha, float beta, int boundary_padding_thickness, int n, float2* A_d,int pitch_a, float2* B_d, int pitch_b, float* r_d, int stride_r);

//void reduce_sum_of_squares_poisson_field_residual_device(F alpha, F beta, int boundary_padding_thickness, int n, F2* A_d,int pitch_a, cudaTextureObject_t B_tex, F* r_d, int stride_r);

void reduce_sum_f32_device(int n, float* r_d, int stride_r);

void reduce_sum_f64_device(int n, double* r_d, int stride_r);

template<class F2>
void restrict(int n, F2* dest, int pitch_dest, F2* src, int pitch_src);

