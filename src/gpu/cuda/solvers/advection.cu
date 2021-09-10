#include <stdio.h>

//static cudaArray* tex_array;
//m_q: Number of vertical interior grid points, k_q: Number of horizontal grid points
__global__
void k_advection_2D_f32(float dt, float dy, float dx, int m_q, int k_q, float2* U, int pitch_u, cudaTextureObject_t Q, float* C, int pitch_c){
	
	const int TILE_WIDTH=8; 
	const int TILE_HEIGHT=8;

	int idy=blockIdx.y*blockDim.y*TILE_HEIGHT+threadIdx.y*TILE_HEIGHT;
	int idx=blockIdx.x*blockDim.x*TILE_WIDTH+threadIdx.x*TILE_WIDTH;
	
	int i=0;

	float2 p;
	C=(float*) ((char*)C+idy*pitch_c);
	U=(float2*) ((char*)U+idy*pitch_u);
	float2* U_ptr=U;
	float* C_ptr=C;

	while (i<m_q){
		for (int i1=0;i1<TILE_HEIGHT;i1++){
			int fy=idy+i1;			
			if ((fy+i)<m_q){
				int j=0;
				while(j<k_q){
				//printf("y:%d\n",fy);	
					for (int i2=0;i2<TILE_WIDTH;i2++){
						int fx=idx+i2;
						if ((fx+j)<k_q){
							//printf("i: %d j: %d y: %d x:%d\n",i,j,fy,fx);
							float2 v=U_ptr[fx+j];
							p.x=(fx+j+1.5f)-(dt*v.x*dx);
							p.y=(fy+i+1.5f)-(dt*v.y*dy);
							float q=tex2D<float>(Q,p.x,p.y);
							C_ptr[fx+j]=q;					
						}		
						else{
							break;
						}
					}					
					j+=gridDim.x*blockDim.x*TILE_WIDTH;	
				}		
			}						
			C_ptr=(float*) ((char*)C_ptr+pitch_c);
			U_ptr=(float2*) ((char*)U_ptr+pitch_u);
		}	
		i+=gridDim.y*blockDim.y*TILE_HEIGHT;
		C_ptr=(float*) ((char*)C+i*pitch_c);
		U_ptr=(float2*) ((char*)U+i*pitch_u);		
	}
}

__global__
void k_advection_2d_f32(float dt, float dy, float dx, int m_q, int k_q, float2* U, int pitch_u, cudaTextureObject_t Q, float2* C, int pitch_c){
	const int TILE_WIDTH=8; 
	const int TILE_HEIGHT=8;

	int idy=blockIdx.y*blockDim.y*TILE_HEIGHT+threadIdx.y*TILE_HEIGHT;
	int idx=blockIdx.x*blockDim.x*TILE_WIDTH+threadIdx.x*TILE_WIDTH;
	
	int i=0;

	float2 p;
	C=(float2*) ((char*)C+idy*pitch_c);
	U=(float2*) ((char*)U+idy*pitch_u);	
	float2* U_ptr=U;
	float2* C_ptr=C;

	while (i<m_q){
		for (int i1=0;i1<TILE_HEIGHT;i1++){
			int fy=idy+i1;			
			if ((fy+i)<m_q){
				int j=0;			
				while(j<k_q){
				//printf("y:%d\n",fy);	
					for (int i2=0;i2<TILE_WIDTH;i2++){
								
						int fx=idx+i2;
						if ((fx+j)<k_q){
							//printf("i: %d j: %d y: %d x:%d\n",i,j,fy,fx);
							float2 v=U_ptr[fx+j];
							p.x=(fx+j+1.5f)-(dt*v.x*dx);// we add 1.5 because of boundary conditions offset, else it would be 0.5
							p.y=(fy+i+1.5f)-(dt*v.y*dx);// we add 1.5 because of boundary conditions offset, else it would be 0.5
							float2 q=tex2D<float2>(Q,p.x,p.y);
							C_ptr[fx+j]=q;					
						}		
						else{
							break;
						}
					}					
					j+=gridDim.x*blockDim.x*TILE_WIDTH;	
				}		
			}						
			C_ptr=(float2*) ((char*)C_ptr+pitch_c);
			U_ptr=(float2*) ((char*)U_ptr+pitch_u);
		}	
		i+=gridDim.y*blockDim.y*TILE_HEIGHT;
		C_ptr=(float2*) ((char*)C+i*pitch_c);
		U_ptr=(float2*) ((char*)U+i*pitch_u);		
	}
}


__host__
void advection_2D_f32_device(float dt, float dy, float dx,  int m_q,  int k_q, float2* U_d, int pitch_u, float* Q_d, int pitch_q, float* C_d, int pitch_c){
	if ((m_q<3) || (k_q<3)){
		return;
	}

	//Create Resource description
	cudaResourceDesc resDesc;
	memset(&resDesc,0,sizeof(resDesc));

	resDesc.resType = cudaResourceTypePitch2D;
	resDesc.res.pitch2D.devPtr=Q_d;
	resDesc.res.pitch2D.width=k_q;
	resDesc.res.pitch2D.height=m_q;
	resDesc.res.pitch2D.pitchInBytes=pitch_q;
	resDesc.res.pitch2D.desc=cudaCreateChannelDesc<float>(); //is equivalent to cudaCreateChannelDesc<float>()
	
	/*
	resDesc.res.pitch2D.desc=cudaCreateChannelDesc(32,32,0,0,cudaChannelFormatKindFloat); //is equivalent to cudaCreateChannelDesc<float2>()
*/
	//Create Texture description
	cudaTextureDesc texDesc;
	memset(&texDesc,0,sizeof(texDesc));
    texDesc.normalizedCoords = false;
	texDesc.filterMode = cudaFilterModeLinear;
	texDesc.readMode=cudaReadModeElementType;
    texDesc.addressMode[0] = cudaAddressModeClamp;
	texDesc.addressMode[1] = cudaAddressModeClamp;

	//Create Texture Object
	cudaTextureObject_t Q_tex;
    cudaError_t error1=cudaCreateTextureObject(&Q_tex, &resDesc, &texDesc, NULL);
	if (error1 !=cudaSuccess){
		printf("Errorcode: %d\n",error1);
	}
	printf("w, h: %d,%d\n",k_q,m_q);
	float* C_ptr=(float*) ((char*)C_d+pitch_c)+1;
	float2* U_ptr=(float2*) ((char*)U_d+pitch_u)+1;
	k_advection_2D_f32<<<dim3(1,1,1),dim3(8,4,1)>>>(dt,dy,dy,m_q-2,k_q-2,U_ptr,pitch_u,Q_tex,C_ptr,pitch_c);


}


