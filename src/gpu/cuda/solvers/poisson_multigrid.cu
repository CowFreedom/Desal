#include <stdio.h>
//Calculates (\laplace p)x=b, whereas A is a finite difference M matrix
//As the structure of (\laplace p) is thereas fixed, no matrix has to be passed
__global__
void k_mg_vc_poisson_2D_f32(float* p, int stride_p, float* b, int stride_b, float* x, int stride_x);


__global__
void k_jacobi_poisson_2D_f32(float weight, float alpha, float beta_inv, int m, int k, cudaTextureObject_t X_old, float2* X_new, int pitch_x, cudaTextureObject_t B){
	//static __shared__ float2 As[BLOCKDIM];
	const int TILE_WIDTH=1;
	const int TILE_HEIGHT=1; //number of elements a thread is setting

	int idy=blockIdx.y*blockDim.y*TILE_HEIGHT+threadIdx.y*TILE_HEIGHT;
	int idx=blockIdx.x*blockDim.x*TILE_WIDTH+threadIdx.x*TILE_WIDTH;
	
	float2* X_ptr=X_new;
	X_ptr=(float2*) ((char*)X_new+pitch_x);	
	int i=0;
	while (i<m){
		for (int i1=0;i1<TILE_HEIGHT;i1++){
			int fy=idy+i1;			
			if ((fy+i)<m){
				int j=0;
				while(j<k){
					for (int i2=0;i2<TILE_WIDTH;i2++){
						int fx=idx+i2;
						if ((fx+j)<k){

							float2 xupper=tex2D<float2>(X_old,fx+j+1.5f,fy+i+1+1.5f);
							float2 xlower=tex2D<float2>(X_old,fx+j+1.5f,fy+i-1+1.5f);
							float2 xright=tex2D<float2>(X_old,fx+j+1+1.5f,fy+i+1.5f);
							float2 xleft=tex2D<float2>(X_old,fx+j-1+1.5f,fy+i+1.5f);						
							
							float2 b=tex2D<float2>(B,fx+j+1.5f,fy+i+1.5f);
							//printf("Valb:(%f,%f)index: %d,%d\n",b.x,b.y, fx+j,fy+i);
							X_ptr[fx+j+1].x=(1.0-weight)*X_ptr[fx+j+1].x+weight*beta_inv*(xlower.x+xupper.x+xleft.x+xright.x+alpha*b.x);	
							X_ptr[fx+j+1].y=(1.0-weight)*X_ptr[fx+j+1].y+weight*beta_inv*(xlower.y+xupper.y+xleft.y+xright.y+alpha*b.y);		
							//X_ptr[fx+j+1].x=b.x;	
							//X_ptr[fx+j+1].y=9;		
							//printf("Valy:%f\n",X_ptr[fx+j].y);							
						}		
						else{
							break;
						}
					}					
					j+=gridDim.x*blockDim.x*TILE_WIDTH;	
				}		
			}						

			X_ptr=(float2*) ((char*)X_ptr+pitch_x);
		}	
		i+=gridDim.y*blockDim.y*TILE_HEIGHT;
		X_ptr=(float2*) ((char*)X_new+(i+1)*pitch_x);	 //check if i+1 is correct
	}
}

__global__
void k_test(cudaTextureObject_t A){
	//float2 b=tex2D<float2>(A,0.5+1,0.5+1);
	//printf("Test: %f,%f",b.x,b.y);
}


//Solves AX=B
__host__
void jacobi_poisson_2D_f32_device(float jacobi_weight, float alpha, float beta_inv, int m, int k,  float2* X_old, int pitch_x_old, float2* X, int pitch_x, cudaTextureObject_t B){
	
	//Create Resource description
	cudaResourceDesc resDesc;
	memset(&resDesc,0,sizeof(resDesc));

	resDesc.resType = cudaResourceTypePitch2D;
	resDesc.res.pitch2D.devPtr=X_old;
	resDesc.res.pitch2D.pitchInBytes=pitch_x_old;
	resDesc.res.pitch2D.width=k;
	resDesc.res.pitch2D.height=m;

	resDesc.res.pitch2D.desc=cudaCreateChannelDesc<float2>(); //is equivalent to cudaCreateChannelDesc(32,32,0,0,cudaChannelFormatKindFloat)

	//Create Texture description
	cudaTextureDesc texDesc;
	memset(&texDesc,0,sizeof(texDesc));
    texDesc.normalizedCoords = false;
	texDesc.filterMode = cudaFilterModeLinear; //change to nearest
	texDesc.readMode=cudaReadModeElementType;
    texDesc.addressMode[0] = cudaAddressModeClamp;
	
	cudaTextureObject_t X_old_tex;

	cudaError_t error1=cudaCreateTextureObject(&X_old_tex, &resDesc, &texDesc, NULL);
	if (error1 !=cudaSuccess){
		printf("Errorcode: %d\n",error1);
	}

	for (int i=0;i<200;i++){
		
		k_jacobi_poisson_2D_f32<<<1,1>>>(jacobi_weight,alpha,beta_inv,m-2,k-2,X_old_tex,X,pitch_x,B);	
		
		float2* temp=X_old;
		X_old=X;
		X=temp;
		size_t pitch_temp=pitch_x;
		pitch_x=pitch_x_old;
		pitch_x_old=pitch_temp;		
		resDesc.res.pitch2D.devPtr=X_old;
		resDesc.res.pitch2D.pitchInBytes=pitch_x_old;
	}

}
/*
__global__
void k_copyVerticalBoundary2D(int m, int k, float* destination, size_t pitch_d, float* source, size_t pitch_s, float* source){
	const int TILE_LENGTH=2; //number of elements a thread is setting

	int idx=blockIdx.y*blockDim.y*TILE_LENGTH+threadIdx.y*TILE_LENGTH;
	int idy=blockIdx.x*blockDim.x*TILE_LENGTH+threadIdx.x*TILE_LENGTH;
	
	//todo rest
}
*/
//AC=B
__host__
void mg_vc_poisson_2D_f32(float alpha, float beta, int m, int k, float2* B_d, int pitch_b, float2* C_d, int pitch_c){
	float beta_inv=1.0/beta;
	int l=1;
	float jacobi_weight=1.0;
	
	//Create Resource description
	cudaResourceDesc resDesc;
	memset(&resDesc,0,sizeof(resDesc));
	resDesc.resType = cudaResourceTypePitch2D;
	resDesc.res.pitch2D.devPtr=B_d;
	resDesc.res.pitch2D.width=k;
	resDesc.res.pitch2D.height=m;
	resDesc.res.pitch2D.pitchInBytes=pitch_b;
	resDesc.res.pitch2D.desc=cudaCreateChannelDesc<float2>(); //is equivalent to cudaCreateChannelDesc(32,32,0,0,cudaChannelFormatKindFloat)

	//Create Texture description
	cudaTextureDesc texDesc;
	memset(&texDesc,0,sizeof(texDesc));
    texDesc.normalizedCoords = false;
	texDesc.filterMode = cudaFilterModeLinear;
	texDesc.readMode=cudaReadModeElementType;
    texDesc.addressMode[0] = cudaAddressModeClamp;

	//Create Texture Object
	cudaTextureObject_t B_tex;
    cudaError_t error1=cudaCreateTextureObject(&B_tex, &resDesc, &texDesc, NULL);
	/*if (error1 !=cudaSuccess){
		printf("Errorcode: %d\n",error1);
	}
	*/
	float2* U_buf; //holding contents of intermediary U results of the various grid sizes
	size_t pitch_u_buf;
	cudaMallocPitch((void**)&U_buf,&pitch_u_buf,2*sizeof(float2)*k,m);
	cudaMemcpy2D(U_buf,pitch_u_buf,C_d,pitch_c,k*sizeof(float2),m,cudaMemcpyDeviceToDevice);
	
	
//k_test<<<1,1>>>(B_tex);
	jacobi_poisson_2D_f32_device(jacobi_weight,alpha,beta_inv,m,k, U_buf,pitch_u_buf,C_d,pitch_c, B_tex);
	//k_jacobi_poisson_2D_f32<<<1,1>>>(jacobi_weight,alpha,beta_inv,m-2,k-2,U_tex,C_ptr,pitch_c,horizontal_offset_c);
	cudaFree(U_buf);
}