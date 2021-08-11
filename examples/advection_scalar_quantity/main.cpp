#include<iostream>
#include "..\..\src\gpu\hostgpu_bindings.h"

template<class T>
void print_scalar_field(T* A, int n, int m){
	for (int i=0;i<n;i++){
		for (int j=0;j<m;j++){
			std::cout<<A[i*n+j]<<" ";
		}
		std::cout<<"\n";
	}	
}

template<class T>
void print_vector_field(T* A, int n, int m){
	for (int i=0;i<n;i++){
		for (int j=0;j<m;j++){
			std::cout<<"("<<A[i*2*n+j*2]<<","<<A[i*2*n+j*2+1]<<") ";
		}
		std::cout<<"\n";
	}	
}


template<class T>
void run_example(T width, T height, int n, int m){
	
	
	T* Q=new T[n*m];
	T* U=new T[2*n*m];
	T* C=new T[n*m];
	T t0=0.0;
	T tend=5;
	T dt=1;
	for(int i=0;i<n*m;i++){
		Q[i]=0;
		U[i*2]=1;
		U[i*2+1]=1;
	}
	
	for (int i=0;i<n;i++){
		for (int j=0;j<m;j++){
			if (j>7){
				//U[i*m*2+j*2]*=-1;
			//	U[i*m*2+j*2+1]*=-1;
				
			}
		}
	}
	
	Q[m+5]=1;
	Q[m+6]=3;
	
	T t=t0+dt;
	std::cout<<"Q:\n";
	print_scalar_field(Q,n,m);	
	std::cout<<"\nU:\n";
	print_vector_field(U,n,m);
	while(t<=tend){
		
		//Inefficient. Keep memory in device
		gpu_advection_2d_f32(height, width,n,m, U,1,2*m, Q, 1,m, dt,C,1, m);
		std::cout<<"\n"<<"t="<<t<<", Q:\n";
	
		std::memcpy(Q,C,sizeof(T)*n*m);
		print_scalar_field(Q,n,m);	
		t+=dt;
	}
	
	delete[] Q;
	delete[] U;
	delete[] C;
}

int main(){
	float width=18.0;
	float height=18.0;
	float x_points=18;
	float y_points=18;
	run_example(width,height,x_points,y_points);
}
