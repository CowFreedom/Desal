#include<iostream>


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
	T tend=0.25;
	T dt=0.125;
	for(int i=0;i<n*m;i++){
		Q[i]=1;
		U[i*2]=1;
		U[i*2+1]=2;
	}
	
	T t=t0;
	std::cout<<"Q:\n";
	print_scalar_field(Q,n,m);	
	std::cout<<"\nU:\n";
	print_vector_field(U,n,m);
	while(t<=tend){
		

		
		std::cout<<"\n"<<"t="<<t<<", Q:\n";
		print_scalar_field(Q,n,m);		
				
		t+=dt;
	}
	
	delete[] Q;
	delete[] U;
	delete[] C;
}

int main(){
	float width=1.0;
	float height=1.0;
	float x_points=32;
	float y_points=32;
	run_example(width,height,x_points,y_points);
}
