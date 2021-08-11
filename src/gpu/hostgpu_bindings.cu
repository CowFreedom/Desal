#include "solvers/advection.cuh"

__host__
void gpu_advection_2d_f32(const float height, const float width,const int m_q, const int k_q, const float* U_h, int stride_row_u, int stride_col_u, const float* Q_h, int stride_row_q, int stride_col_q, float dt, float* C_h, int stride_row_c, int stride_col_c)
{
	advection_2d_f32(height,width,m_q,k_q,U_h,stride_row_u,stride_col_u,Q_h,stride_row_q,stride_col_q,dt,C_h,stride_row_c,stride_col_c);		
}