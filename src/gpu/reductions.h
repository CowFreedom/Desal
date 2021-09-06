//AX=B
template<unsigned int THREADS_X_PER_BLOCK,unsigned int THREADS_Y_PER_BLOCK>
__global__
void k_reduce_sos_fivepoint_stencil_float2(float alpha_inv, float beta, float boundary_offset, cudaTextureObject_t A,cudaTextureObject_t B, float* r, int stride_r);

template<unsigned int THREADS_X_PER_BLOCK, class F>
__global__
void k_reduce_sum(int n, float* r, int stride_r);