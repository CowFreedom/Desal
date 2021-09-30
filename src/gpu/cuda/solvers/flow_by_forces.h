#pragma once

namespace desal{
	
	namespace cuda{
		
		template<class F, class F2>
		void k_add_forces(F dt, int boundary_padding_thickness, int m, int k,  F2* A, size_t pitch_a, F2* F, size_t pitch_f, F2* C, size_t pitch_c){
			m-=2*boundary_padding_thickness;
			k-=2*boundary_padding_thickness;
			
			A=(F2*)((char*)A+boundary_padding_thickness*pitch_a)+boundary_padding_thickness;
			F=(F2*)((char*)F+boundary_padding_thickness*pitch_f)+boundary_padding_thickness;
			C=(F2*)((char*)C+boundary_padding_thickness*pitch_c)+boundary_padding_thickness;
			for (int i=blockIdx.y*blockDim.y+threadIdx.y;i<m;i+=gridDim.y*blockDim.y){
				A=(F2*)((char*)A+i*pitch_a);
				F=(F2*)((char*)F+i*pitch_f);
				C=(F2*)((char*)C+i*pitch_c);
				
				for (int j=blockIdx.x*blockDim.x+threadIdx.x;j<k;j+=gridDim.x*blockDim.x){
					F2 v=F[j];
					v.x=A[j].x+dt*v.x;
					v.y=A[j].y+dt*v.y;
					A[j]=v;
				}
			}		
		}

		template<class F, class F2>
		void add_forces(F dt, int boundary_padding_thickness, int m, int k, F2* A_d, size_t pitch_a, F2* F_d, size_t pitch_f, F2* C_d, size_t pitch_c){
			float threads_x=32;
			float threads_y=32;
			dim3 threads=dim3(threads_x,threads_y,1);
			dim3 blocks=dim3(ceil(k/threads_x),ceil(m/threads_y),1);
			k_add_forces(dt, boundary_padding_thickness, m, k, A_d, pitch_a, F_d, pitch_f, C_d, pitch_c);				
		}
	}
}