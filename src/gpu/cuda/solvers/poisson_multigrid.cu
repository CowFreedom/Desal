#include <stdio.h>
#include "../reductions.h"
#include "../transformations.h"

//Calculates (\laplace p)x=b, whereas A is a finite difference M matrix
//As the structure of (\laplace p) is thereas fixed, no matrix has to be passed
__global__
void k_mg_vc_poisson_2D_f32(float* p, int stride_p, float* b, int stride_b, float* x, int stride_x);

//m: height of interior points k: width of interior points
template<class F, class F2>
__global__
void k_jacobi_poisson_2D(F weight, F alpha, F beta_inv, int boundary_padding_thickness, int n, cudaTextureObject_t X_old, F2* X_new, int pitch_x, cudaTextureObject_t B){
printf("kernel: a%f b %f\n",alpha,beta_inv);
	n-=2*boundary_padding_thickness;
	int idy=blockIdx.y*blockDim.y+threadIdx.y;
	int idx=blockIdx.x*blockDim.x+threadIdx.x;
	
	float2* X_ptr=X_ptr=(float2*) ((char*)X_new+(idy+boundary_padding_thickness)*pitch_x);	

	for(int i=idy;i<n;i+=gridDim.y*blockDim.y){
			
		for(int j = idx; j<n; j+=gridDim.x*blockDim.x){
			float2 x=tex2D<float2>(X_old,j+boundary_padding_thickness+0.5,i+boundary_padding_thickness+0.5);
			float2 xupper=tex2D<float2>(X_old,j+boundary_padding_thickness+0.5,i+1+boundary_padding_thickness+0.5);
			float2 xlower=tex2D<float2>(X_old,j+boundary_padding_thickness+0.5,i-1+boundary_padding_thickness+0.5);
			float2 xright=tex2D<float2>(X_old,j+1+boundary_padding_thickness+0.5,i+boundary_padding_thickness+0.5);
			float2 xleft=tex2D<float2>(X_old,j-1+boundary_padding_thickness+0.5,i+boundary_padding_thickness+0.5);						
			float2 b=tex2D<float2>(B,j+boundary_padding_thickness+0.5,i+boundary_padding_thickness+0.5);
		//	printf("Val:(%f,%f)index: %d,%d\n",x.x,x.y, i,j);
			X_ptr[j+boundary_padding_thickness].x=(1.0-weight)*x.x+weight*beta_inv*(xlower.x+xupper.x+xleft.x+xright.x+alpha*b.x);	
			X_ptr[j+boundary_padding_thickness].y=(1.0-weight)*x.y+weight*beta_inv*(xlower.y+xupper.y+xleft.y+xright.y+alpha*b.y);	
		//	printf("X_ptr: %f Valy:%f\n",X_ptr[j+boundary_padding_thickness].x,x.x);									
		}
		X_ptr=(float2*) ((char*)X_ptr+pitch_x);	 //check if i+1 is correct	
	}
}

__global__
void k_test(cudaTextureObject_t A){
	//float2 b=tex2D<float2>(A,0.5+1,0.5+1);
	//printf("Test: %f,%f",b.x,b.y);
}

__global__
void print_vector_field_k2(int m,int k, float2* M, int pitch,char name){
	printf("%c:\n",name);
	for (int i=0;i<m;i++){
		float2* current_row=(float2*)((char*)M + i*pitch);
		for (int j=0;j<k;j++){
			printf("(%.1f,%.1f) ",current_row[j].x,current_row[j].y);
		}
		printf("\n");
	}	
}

//Solves AX=B
template<class F, class F2>
__host__
void jacobi_poisson_2D_device(F jacobi_weight, F alpha, F beta, int boundary_padding_thickness, int n, F2* X_old, int pitch_x_old, F2* X, int pitch_x, cudaTextureObject_t B, int rounds, float2* B_d, int pitch_b){
	F beta_inv=1.0/beta;
	//Create Resource description
	cudaResourceDesc resDesc;
	memset(&resDesc,0,sizeof(resDesc));

	resDesc.resType = cudaResourceTypePitch2D;
	resDesc.res.pitch2D.devPtr=X_old;
	resDesc.res.pitch2D.pitchInBytes=pitch_x_old;
	resDesc.res.pitch2D.width=n;
	resDesc.res.pitch2D.height=n;

	resDesc.res.pitch2D.desc=cudaCreateChannelDesc<F2>(); //cudaCreateChannelDesc(32,32,0,0,cudaChannelFormatKindFloat) is equivalent cudaCreateChannelDesc<float2>()

	//Create Texture description
	cudaTextureDesc texDesc;
	memset(&texDesc,0,sizeof(texDesc));
    texDesc.normalizedCoords = false;
	texDesc.filterMode = cudaFilterModeLinear; //change to nearest
	texDesc.readMode=cudaReadModeElementType;
    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.addressMode[1] = cudaAddressModeClamp;
	
	cudaTextureObject_t X_old_tex;

	cudaError_t error1=cudaCreateTextureObject(&X_old_tex, &resDesc, &texDesc, NULL);
	if (error1 !=cudaSuccess){
		printf("Errorcode: %d\n",error1);
	}
	for (int i=0;i<rounds;i++){
		print_vector_field_k2<<<1,1>>>(n,n,X_old,pitch_x_old,'O');

		F* test;
		cudaMalloc((void**) &test, sizeof(F)*100);
		reduce_sum_of_squares_poisson_field_residual_f32_device(alpha,beta,boundary_padding_thickness, n,X_old,pitch_x_old, B_d, pitch_b, test, 1);	
		
		F residual;
		cudaMemcpy(&residual,test,sizeof(F)*1,cudaMemcpyDeviceToHost);
		printf("mg vc poisson 2d residual: %f\n",residual);
		cudaFree(test);
		
	//	printf("S1: a%f b %f\n",alpha,beta_inv);
		k_jacobi_poisson_2D<F,F2><<<1,1>>>(jacobi_weight,alpha,beta_inv,boundary_padding_thickness,n,X_old_tex,X,pitch_x,B);	
		print_vector_field_k2<<<1,1>>>(n,n,X,pitch_x,'X');
		resDesc.res.pitch2D.devPtr=X;
		resDesc.res.pitch2D.pitchInBytes=pitch_x;

		k_jacobi_poisson_2D<F,F2><<<1,1>>>(jacobi_weight,alpha,beta_inv,boundary_padding_thickness,n,X_old_tex,X_old,pitch_x_old,B);			
		resDesc.res.pitch2D.devPtr=X_old;
		resDesc.res.pitch2D.pitchInBytes=pitch_x_old;
	
	}
}

//AC=B
template<class F, class F2>
__host__
void mg_vc_poisson_2D_device(F alpha, F beta, int boundary_padding_thickness, int n, F2* B_d, int pitch_b, F2* C_d, int pitch_c, F2* C_buf, int pitch_c_buf, F2* r_buf, int pitch_r_buf, F jacobi_weight, int jacobi_rounds, int multigrid_stages){
	
	
	//Create Resource description
	cudaResourceDesc resDesc;
	memset(&resDesc,0,sizeof(resDesc));
	resDesc.resType = cudaResourceTypePitch2D;
	resDesc.res.pitch2D.devPtr=B_d;
	resDesc.res.pitch2D.width=n;
	resDesc.res.pitch2D.height=n;
	resDesc.res.pitch2D.pitchInBytes=pitch_b;
	resDesc.res.pitch2D.desc=cudaCreateChannelDesc<F2>(); 

	//Create Texture description
	cudaTextureDesc texDesc;
	memset(&texDesc,0,sizeof(texDesc));
    texDesc.normalizedCoords = false;
	texDesc.filterMode = cudaFilterModeLinear;
	texDesc.readMode=cudaReadModeElementType;
    texDesc.addressMode[0] = cudaAddressModeClamp;
	texDesc.addressMode[1] = cudaAddressModeClamp;

	//Create Texture Object
	cudaTextureObject_t B_tex;
    cudaError_t error1=cudaCreateTextureObject(&B_tex, &resDesc, &texDesc, NULL);
	/*if (error1 !=cudaSuccess){
		printf("Errorcode: %d\n",error1);
	}
	*/
	

	jacobi_poisson_2D_device<F,F2>(jacobi_weight,alpha,beta,boundary_padding_thickness,n, C_buf,pitch_c_buf,C_d,pitch_c, B_tex,jacobi_rounds,B_d,pitch_b);
	transform_entries_into_square_residuals_device<float,float2>(alpha,beta, boundary_padding_thickness, n,n, C_buf, pitch_c_buf, B_d, pitch_b, r_buf, pitch_r_buf); //TODO C_buf should be equal to C at this stage
	
	for (int stage=1;stage<multigrid_stages;i++){
		//interpolate previous result
		//Solve new system
		
	}
	
	for (int stage=multigrid_stages-1;stage>=0;i--){
		//prolongate previous result
		//Correct error
		Solve new System (relax)
		
	}
	
}

__host__
void mg_vc_poisson_2D_f32_device(float alpha, float beta, int boundary_padding_thickness, int n, float2* B_d, int pitch_b, float2* C_d, int pitch_c){

	int jacobi_rounds=20;
	int multigrid_stages=1;
	float jacobi_weight=1.0;
	
	float2* C_buf; //holding contents of intermediary U results of the various grid sizes
	float2* r_buf;

	size_t pitch_c_buf;
	size_t pitch_r_buf;
	
	cudaMallocPitch((void**)&C_buf,&pitch_c_buf,static_cast<size_t>((1.34*sizeof(float2)*n)),n);
	cudaMemcpy2D(C_buf,pitch_c_buf,C_d,pitch_c,n*sizeof(float2),n,cudaMemcpyDeviceToDevice);
	
	cudaMallocPitch((void**)&r_buf,&pitch_r_buf,static_cast<size_t>(1.34*sizeof(float2)*n),n);
	//print_vector_field_k2<<<1,1>>>(static_cast<size_t>(1.34*n),n,C_buf,pitch_c_buf,'Z');
	mg_vc_poisson_2D_device<float,float2>(alpha, beta, boundary_padding_thickness, n, B_d, pitch_b, C_d, pitch_c, C_buf, pitch_c_buf, r_buf, pitch_r_buf, jacobi_weight,jacobi_rounds,multigrid_stages);
	
	
	cudaFree(C_buf);
	cudaFree(r_buf);
	
}