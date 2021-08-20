#include<iostream>
#include "..\..\src\gpu\hostgpu_bindings.h"

//Prints the scalar quantity field
template<class T>
void print_scalar_field(T* A, int n, int m){
	for (int i=0;i<n;i++){
		for (int j=0;j<m;j++){
			std::cout<<A[i*n+j]<<" ";
		}
		std::cout<<"\n";
	}	
}

//prints the vectorial flow field
template<class T>
void print_vector_field(T* A, int n, int m){
	for (int i=0;i<n;i++){
		for (int j=0;j<m;j++){
			std::cout<<"("<<A[i*2*n+j*2]<<","<<A[i*2*n+j*2+1]<<") ";
		}
		std::cout<<"\n";
	}	
}

//runs the example
template<class T>
void run_example(T width, T height, int n, int m){
	
	T* Q=new T[n*m]; //quantity vector
	T* U=new T[2*n*m]; //flow field vector
	T* C=new T[n*m]; // stores results of the advected quantity field Q
	
	T t0=0.0; //start time simulation
	T tend=0.1; //end time simulation
	T dt=0.1; //step size of the solver
	T dx=width/m;
	T dy=height/n;
	for(int i=0;i<n*m;i++){
		Q[i]=0;
		U[i*2]=1;
		U[i*2+1]=2;
	}
	
	for (int i=0;i<n;i++){
		for (int j=0;j<m;j++){
			if (j>7){
				//U[i*m*2+j*2]*=-1;
				//U[i*m*2+j*2+1]*=-1;		
			}
		}
	}
	
	Q[m+5]=1;
	//Q[m+6]=3;
	
	T t=t0+dt;
	std::cout<<"Q:\n";
	print_scalar_field(Q,n,m);	
	std::cout<<"\nU:\n";
	print_vector_field(U,n,m);
	
	while(t<=tend){
		
		//Inefficient. Keep memory in device
		gpu_advection_2d_f32(dt,dx,dy,n,m, U,2*m, Q, m, C, m);
		
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
	float width=18.0; //width of the rectangular grid
	float height=18.0; //height of the rectangular grid
	float x_points=18; //number of gridpoints (including boundary points) in x direction 
	float y_points=18; //number of gridpoints (including boundary points) in y direction
	run_example(width,height,x_points,y_points);
}
