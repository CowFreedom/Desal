#pragma once

__host__
void mg_vc_poisson_2D_f32(float alpha, float beta, int m, int k, float2* B_d, int pitch_b, float2* C_d, int pitch_c);