#include <stdio.h>
//This advects a scalar quantity. Assumes padding (with e.g. boundary conditions around the border)
__global__
void k_advection_2d_f32(const float height, const float width,const int m_q, const int k_q, const float* U, int stride_row_u, int stride_col_u, const float* Q, int stride_row_q, int stride_col_q, float dt, float* C, int stride_row_c, int stride_col_c){
	const int TILE_WIDTH=8; 
	const int TILE_HEIGHT=8;
	const int THREADS_PER_BLOCK_X=2;
	const int THREADS_PER_BLOCK_Y=2;
	int tx=threadIdx.x;
	int ty=threadIdx.y;
	int bx=blockIdx.x;
	int by=blockIdx.y;
	//int off_uy=by*TILE_HEIGHT*THREADS_PER_BLOCK_Y+ty*TILE_HEIGHT;
	//int off_ux=2*bx*TILE_WIDTH*THREADS_PER_BLOCK_X+2*tx*TILE_WIDTH;
	
	int off_u=by*TILE_HEIGHT*THREADS_PER_BLOCK_Y*stride_col_u+ty*TILE_HEIGHT*stride_col_u+2*bx*TILE_WIDTH*THREADS_PER_BLOCK_X*stride_row_u+2*tx*TILE_WIDTH*stride_row_u;
//	int off_q=by*TILE_HEIGHT*THREADS_PER_BLOCK_Y*stride_col_q+ty*TILE_HEIGHT*stride_col_q+bx*TILE_WIDTH*THREADS_PER_BLOCK_X*stride_row_q+tx*TILE_WIDTH*stride_row_q;
	int off_cx=bx*TILE_WIDTH*THREADS_PER_BLOCK_X*stride_row_c+tx*TILE_WIDTH*stride_row_c;
	int off_cy=by*TILE_HEIGHT*THREADS_PER_BLOCK_Y*stride_col_c+ty*TILE_HEIGHT*stride_col_c;
	int off_c=off_cx+off_cy;
	
	//origin in world coordinates
	float cell_lengthx=width/k_q;
	float cell_lengthy=height/m_q;
	float ox=(bx*TILE_WIDTH*THREADS_PER_BLOCK_X+tx*TILE_WIDTH+1.5)*cell_lengthx;
	float oy=(by*TILE_HEIGHT*THREADS_PER_BLOCK_Y+ty*TILE_HEIGHT+1.5)*cell_lengthy;
	float inv_interior_width=1.0/(width-2*cell_lengthx);
	float inv_interior_height=1.0/(height-2*cell_lengthy);
	//float inv_width=1.0/(width);
	//float inv_height=1.0/(height);	
	for (int i=0;i<TILE_HEIGHT;i++){
		for (int j=0;j<TILE_WIDTH; j++){
		
			//Determine current location in world coordinates (respective to rectangle)
			float ouy=oy+(i)*cell_lengthy;
			float oux=ox+(j)*cell_lengthx;
			
			//Approximate wave path in a linear manner
			float dux=U[off_u+i*stride_col_u+j*2*stride_row_u];
			float duy=U[off_u+i*stride_col_u+j*2*stride_row_u+1];
			
			float ux=oux-dt*dux;
			float uy=ouy-dt*duy;
			
			//Convert result to grid indices
			int gx=((ux-1*cell_lengthx)*inv_interior_width*(k_q-2))-0.5; //I substract the minus 0.5 in order to round numbers toward the left
			int gy=((uy-1*cell_lengthy)*inv_interior_height*(m_q-2))-0.5;
			//printf("gy,gx: %d,%d, cy,cx: %d,%d und %f,%f und %f\n",gy,gx,by*TILE_HEIGHT*THREADS_PER_BLOCK_Y+ty*TILE_HEIGHT+i,off_cx+j,(uy-1*cell_lengthy)*inv_interior_height*(m_q-2),(ux-1*cell_lengthx)*inv_interior_width*(k_q-2),ux-1*cell_lengthx);
			gx=(gx < 0)? -1 : (gx >= k_q)? k_q-1 : gx;
			gy=(gy < 0)? -1 : (gy >= m_q)? m_q-1 : gy;			
			//Interpolate 
			
			//int gx=round(bx*TILE_WIDTH*THREADS_PER_BLOCK_X+tx*TILE_WIDTH+j-gux*k_q);
			//gx=(gx < 0)? 0 : (gx >= k_q)? k_q-1 : gx;
			//int gy=round(by*TILE_HEIGHT*THREADS_PER_BLOCK_Y+ty*TILE_HEIGHT+i-guy*m_q);
			//gy=(gy < 0)? 0 : (gy >= m_q)? m_q-1 : gy;
			//printf("off_q: %d, newq: %d,%d \n",+j*stride_row_q,gx,gy);
			//C[off_c+i*stride_col_c+j*stride_row_c]=Q[off_q+i*stride_col_q+j*stride_row_q+gx+gy];
			//printf("Q[%d,%d]=%f\n",off_q+i*stride_col_q,j*stride_row_q,Q[off_q+i*stride_col_q+j*stride_row_q]);
			C[off_c+i*stride_col_c+j*stride_row_c]=Q[gy*stride_col_q+gx*stride_row_q];
		}
	}
/*
	float inv_width=1.0/width;
	float inv_height=1.0/height;
	for (int i=0;i<TILE_HEIGHT;i++){
		for (int j=0;j<TILE_WIDTH; j++){
			float wux=dt*U[off_u+i*stride_col_u+j*2*stride_row_u];
			float wuy=dt*U[off_u+i*stride_col_u+j*2*stride_row_u+1];
			float gux=wux*inv_width;
			float guy=wuy*inv_height;
			int gx=round(bx*TILE_WIDTH*THREADS_PER_BLOCK_X+tx*TILE_WIDTH+j-gux*k_q);
			//gx=(gx < 0)? 0 : (gx >= k_q)? k_q-1 : gx;
			int gy=round(by*TILE_HEIGHT*THREADS_PER_BLOCK_Y+ty*TILE_HEIGHT+i-guy*m_q);
			//gy=(gy < 0)? 0 : (gy >= m_q)? m_q-1 : gy;
			//printf("off_q: %d, newq: %d,%d \n",+j*stride_row_q,gx,gy);
			//C[off_c+i*stride_col_c+j*stride_row_c]=Q[off_q+i*stride_col_q+j*stride_row_q+gx+gy];
			//printf("Q[%d,%d]=%f\n",off_q+i*stride_col_q,j*stride_row_q,Q[off_q+i*stride_col_q+j*stride_row_q]);
			C[off_c+i*stride_col_c+j*stride_row_c]=Q[gy*stride_col_q+gx*stride_row_q];
		}
	}
*/
}

__host__
void advection_2d_f32(const float height, const float width,const int m_q, const int k_q, const float* U_h, int stride_row_u, int stride_col_u, const float* Q_h, int stride_row_q, int stride_col_q, float dt, float* C_h, int stride_row_c, int stride_col_c){
if ((m_q<3) || (k_q<3)){
		return;
	}
	
	float* Q_d;
	float* U_d;
	float* C_d;
	int sizeQ=sizeof(float)*m_q*k_q;
	int sizeU=2*sizeQ;
	int sizeC=sizeQ;
	
	cudaMalloc((void**)&Q_d, sizeQ);
	cudaMalloc((void**)&U_d,sizeU);
	cudaMalloc((void**)&C_d,sizeC);
	printf("sizeQ: %d\n",sizeQ);
	cudaError_t copy1=cudaMemcpy((void*) Q_d, (void*) Q_h, sizeQ,cudaMemcpyHostToDevice);
	cudaError_t copy2=cudaMemcpy((void*) U_d, (void*) U_h, sizeU,cudaMemcpyHostToDevice);
	cudaError_t copy3=cudaMemcpy((void*) C_d, (void*) Q_h, sizeC,cudaMemcpyHostToDevice);	
	//cudaMemset((void*) C_d,0,sizeC);
	if ((copy1==cudaSuccess) && (copy2==cudaSuccess) && (copy3==cudaSuccess)){
		float bsmx=2*8; //blocksize x
		float bsmy=2*8; //blocksize y	
		float tbx=2; //threadsize x
		float tby=2; //threadsize y		
		dim3 threadLayout=dim3(tbx,tby,1);
		dim3 grid=dim3(ceil((m_q-2)/bsmx),ceil((k_q-2)/bsmy),1);
		k_advection_2d_f32<<<grid,threadLayout>>>(height,width,m_q,k_q,U_d+stride_col_u+2*stride_row_u,stride_row_u,stride_col_u,Q_d+stride_col_q+stride_row_q,stride_row_q,stride_col_q,dt,C_d+stride_col_c+stride_row_c,stride_row_c,stride_col_c);	
		
		cudaError_t copy4=cudaMemcpy((void*)C_h,(void*)C_d,sizeC,cudaMemcpyDeviceToHost);
		if (copy4!=cudaSuccess){
			printf("%s\n",cudaGetErrorString(copy4));
		}
		
	}
	else{
		printf("Error copying value to device in k_advection_2d_f32\n");
		if (copy1!=cudaSuccess){
			printf("Es war auc copy1 %s\n",cudaGetErrorString(copy1));
		}
	}
	
	cudaFree(Q_d);
//	cudaFree(U_d);
//	cudaFree(C_d);
}
