#pragma once


__host__
void advection_2D_f32_device(float dt, float dy, float dx,  int m_q,  int k_q,  float2* U_h, int pitch_u,  float* Q_h, int pitch_q, float* C_h, int pitch_c);