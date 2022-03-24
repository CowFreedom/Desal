/*This module contains math routines useful for all other modules*/

module;
#include <optional>
#include <limits>
#include <cmath>
export module desal.math;

namespace desal{
	namespace math{
		
		/*This is the Lambert function. Halley's Method is used to evaluate the Lambert function.*/
		export template<std::floating_point T>
		std::optional<T> lambert(T x, T tol=1e-100, int max_iterations=100, T a_n=1.0){
			
			T exp_a;
			T a_prev;
			T term1;
			int iter=0;
	//		std::cout<<x<<"\n";
	//		std::cin.get();
			do{
				exp_a=std::exp(a_n);
				term1=a_n*exp_a-x;
				a_prev=a_n;
				a_n-=(term1)/(exp_a*(a_n+1.0)-(((a_n+2.0)*(term1))/(2.0*a_n+2.0)));
				//std::cout<<std::abs(a_n-a_prev)<<"\n";
				iter++;
			}
			while (std::abs(a_n-a_prev)>tol && iter<max_iterations);
			if (std::isnan(a_n)||std::isinf(a_n) || iter>max_iterations){
					return {};
			}
			else{
			//	std::cout<<"res:"<<a_n<<"\n";
				return {a_n};
			}
		}
		
	}
	
}
