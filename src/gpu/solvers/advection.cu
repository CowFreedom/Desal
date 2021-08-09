//This advects a scalar quantity
__global__
void k_advection_2d_f32(const float height, const float width,const int m_q, const int k_q, const float* U, int stride_row_u, int stride_col_u, const float* Q, int stride_row_q, int stride_col_q, float dt, float* C, int stride_row_c, int stride_col_c){
	const int TILE_WIDTH=16; 
	const int TILE_HEIGHT=16;
	const int BLOCK_THREADS_X=32;
	const int BLOCK_THREADS_Y=32;
	int tx=threadIdx.x;
	int ty=threadIdx.y;
	int bx=blockIdx.x;
	int by=blockIdx.y;
	int off_u=by*TILE_HEIGHT*BLOCK_THREADS_Y*stride_col_u+2*bx*TILE_WIDTH*BLOCK_THREADS_X+ty*BLOCK_THREADS_Y*stride_col_u+2*tx*TILE_WIDTH*BLOCK_THREADS_X*stride_col_u;
	int off_q=by*TILE_HEIGHT*BLOCK_THREADS_Y*stride_col_q+bx*TILE_WIDTH*BLOCK_THREADS_X+ty*BLOCK_THREADS_Y*stride_col_q+tx*TILE_WIDTH*BLOCK_THREADS_X*stride_col_q;	
	int off_cx=bx*TILE_WIDTH*BLOCK_THREADS_X+tx*TILE_WIDTH*BLOCK_THREADS_X*stride_col_c;
	int off_cy=by*TILE_HEIGHT*BLOCK_THREADS_Y*stride_col_c+y*BLOCK_THREADS_Y*stride_col_c;
	int off_c=off_cx+off_cy;

	float inv_width=1.0/width;
	float inv_height=1.0/height;
	for (int i=0;i<TILE_HEIGHT;i++){
		for (int j=0;j<TILE_WIDTH; j++){
			float wux=dt*U[off_u+i*stride_col_u+j*2*stride_row_u];
			float wuy=dt*U[off_u+i*stride_col_u+j*2*stride_row_u+1];
			float gux=ux*inv_width;
			float guy=uy*inv_height;
			int gx=off_cx+gux*k;
			gx=(gx < 0)? 0 : (gx >= k_q)? k_q : gx;
			int gy=off_gy+guy*m;
			gy=(gy < 0)? 0 : (gy >= m_q)? kW_q : gy;
			C[off_c+i*stride_col_c+j*stride_row_c]=Q[off_q+i*stride_col_q+j*stride_row_q+gx+gy];
		}
	}
}

__host__
void advection_2d_f32(const float height, const float width,const int m_q, const int k_q, const float* U_h, int stride_row_u, int stride_col_u, const float* Q_h, int stride_row_q, int stride_col_q, float dt, float* C_h, int stride_row_c, int stride_col_c){
if ((n_q==0) || (m_q==0)){
		return;
	}
	
	float* Q_d;
	float* U_d;
	float* C_d;
	int sizeQ=m_q*k_q;
	int sizeU=2*sizeQ;
	int sizeC=sizeQ;
	cudaMalloc((void**)&Q_d, sizeof(float)*sizeQ);
	cudaMalloc((void**)&U_d,sizeof(float)*sizeU);
	cudaMalloc((void**)&C_d,sizeof(float)*sizeC);

	cudaError_t copy1=cudaMemcpy((void*) Q_d, (void*) Q_h, sizeQ,cudaMemcpyHostToDevice);
	cudaError_t copy2=cudaMemcpy((void*) U_d, (void*) U_h, sizeU,cudaMemcpyHostToDevice);
	cudaError_t copy3=cudaMemcpy((void*) C_d, (void*) C_h, sizeC,cudaMemcpyHostToDevice);	
	
	
	if ((copy1==cudaSuccess) && (copy2==cudaSuccess) && (copy3==cudaSuccess)){
		float bsmx=32; //blocksize x
		float bsmy=32; //blocksize y	
		float tbx=32; //threadsize x
		float tby=32; //threadsize y		
		dim3 threadLayout=dim3(tbx,tby,1);
		dim3 grid=dim3(ceil(m_q/bsmx),ceil(k_q/bsmy),1);
		k_advection_2d_f32<<<grid,threadLayout>>>(height,width,m_q,k_q,U_d,stride_row_u,stride_col_u,Q_d,stride_row_q,stride_col_q,dt,C_d,stride_row_c,stride_col_c);	
		
		cudaMemcpy((void*)C_h,(void*)C_d,sizeA,cudaMemcpyDeviceToHost);
		
	}
	else{
		printf("Error copying value to device in k_advection_2d_f32\n");
	}
	cudaFree(Q_d);
	cudaFree(U_d);
	cudaFree(C_d);
}
