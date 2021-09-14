#pragma once

__host__
void mg_vc_poisson_2D_f32_device(float alpha, float beta, int boundary_padding_thickness, int n, float2* B_d, int pitch_b, float2* C_d, int pitch_c);