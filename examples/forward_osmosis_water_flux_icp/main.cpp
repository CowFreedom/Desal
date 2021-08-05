/** \file main.cpp
 * This example estimates water flux subject to varying changing membrane parameters.
 * It mirrors the calculations done in e.g. pg. 22 in "Forward Osmosis Fundamentals and Applications" by Ho Kyong Shon et al.
 */
#include <iostream>
#include<optional>
import desal.forward_osmosis.flux_models;
import desal.math;

enum class FluxMode{
	ICPDilutive,
	ICPConcentrative
};

//Vary membrane structural parameter S 
void change_membrane_structural_parameter(double D, double p_draw, double p_feed, double A, double B, int n, double* S, double* Jv, FluxMode mode){

	
	for (int i=0;i<n;i++){
		S[i]=(static_cast<double>(i)/n)*1e-3;
		double K_m=D/S[i];
		std::optional<double> result;
		switch (mode){
			case FluxMode::ICPDilutive: {
				 result=desal::fo::water_flux_dilutive_icp(A,p_draw, p_feed, B,S[i],D);
				 break;
			}
			case FluxMode::ICPConcentrative:{
				 result=desal::fo::water_flux_concentrative_icp(A,p_draw, p_feed, B,S[i],D);
				 break;
			}
			
		}
		
		if (result){
			Jv[i]=(*result);
		}
		else{
			std::cerr<<"Error in calculating the flux\n";
		}
		
	}
}

int main(){
	int n=200; //number of values
	double* Jv=new double[n];
	double* S=new double[n];
	
	double D=1.3e-9;
	double p_draw=99.9e+5;
	double p_feed=0.45e+5;

	double A=5e-12;
	double B=1e-7;
//	std::cout<<*desal::math::lambert(-0.34);
	FluxMode mode=FluxMode::ICPConcentrative;
	
	change_membrane_structural_parameter(D,p_draw,p_feed,A,B, n , S,Jv, mode);
	
	
		switch (mode){
			case FluxMode::ICPDilutive: {
				 std::cout<<"Water flux (in liters per m^2 per hour) in a dilutive ICP model (membrane AL:FS configuration) after changing membrane structural parameter S (in mm)\n";
				 break;
			}
			case FluxMode::ICPConcentrative:{
				 std::cout<<"Water flux (in liters per m^2 per hour) in a concentrative ICP model (membrane AL:DS configuration) after changing membrane structural parameter S (in mm)\n";
				 break;
			}
			
		}
			
	for (int i=0;i<n;i++){
		std::cout<<Jv[i]*1000*3600<<"  ";
	}
	std::cout<<"\n";
	
	delete[] Jv;
	delete[] S;
}
