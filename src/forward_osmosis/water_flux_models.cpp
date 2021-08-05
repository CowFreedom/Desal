/** \file flux_models.cpp
 * This module contains models estimating water and solute flux in forward osmosis models. These models
 * are useful to give upper bounds in flux.
 * Many of the models can be found in the book "Forward Osmosis Fundamentals and Applications" by Ho Kyong Shon et al.
 */
module;
#include <iostream>
#include <optional>
#include <cmath>
export module desal.forward_osmosis.flux_models;

import desal.math;

namespace desal{
	namespace fo{

		/*! Estimates forward osmosis water flux through a membrane in an internal concentration polarization (ICP) model. Assumes membrane in AL:DS configuration, therefore
		 * concentrative ICP is encountered.
		 * For more information see Lee et al. 1981 or table 2.1 in "Forward Osmosis Fundamentals and Applications" by Ho Kyong Shon et al.
		@param[in] water_permeability Water permeability coefficient of the membrane (possible units: meter per second per pascal)
		@param[in] pi_draw Osmotic pressure from the draw side (possible units: pascal)
		@param[in] pi_feed Osmotic pressure from the feed side (possible units: pascal)
		@param[in] solute_permeability Permeability of the membrane for solute flux coming from draw to feed (possible units: meter per second)
		@param[in] structural_coef Strutural coefficient of membrane, defines in Equation A10 in "Forward Osmosis Fundamentals and Applications" by Ho Kyong Shon et al. (possible units: meters)
		@param[in] diffusion_coef Diffusion coefficient of solute (possible units: square meters per second)	
		@param[in] tol Tolerance for root finding of Lambert function
		@param[in] max_iterations Maximum iterations in iterative root finding of Lambert function
		@param[in] a_n Initial value in iterative root finding of Lambert function
		\return Optional Water flux in m^3 per m^2 per second, packed in an optional. If Lamberts Function does not find a root, then value is empty
		*/		 
		export template<std::floating_point T>
		std::optional<T> water_flux_concentrative_icp(T water_permeability, T pi_draw, T pi_feed, T solute_permeability, T structural_coef, T diffusion_coef, T tol=1e-100, int max_iterations=100, T a_n=1.0){

			T D=water_permeability*pi_draw+solute_permeability;;
			T E=-1.0;	
			T res;
			
			//Use Halley's Method if there is no zero in the K_M denominator
			if (structural_coef != 0.0){			
				T K_m=diffusion_coef/structural_coef;
				T B=-1.0;
				T C=K_m;
				T A=-K_m*std::log(water_permeability*pi_feed+solute_permeability);
				T F=((B)/(C*E))*std::exp((B*D-A*E)/(C*E));	
				auto lambert_value=desal::math::lambert(F,tol,max_iterations,a_n);
				if (lambert_value){
					res=(C/B)* (*lambert_value)-(D/E);
				}
				else{
					return {};
				}				
			}
			//Without the linear term (B=0.0) we can solve for the root explicitly
			else{
				T C=diffusion_coef;
				T A=-diffusion_coef*std::log(water_permeability*pi_feed+solute_permeability);				
				res=(std::exp(-A/C)-D)/E;
			}
			
	//		std::cout<<"Nullstellenauswertung:"<<A+B*res+C*std::log(D+E*res)<<"\n";
			return {res};
		}

		/*! Estimates forward osmosis water flux through a membrane in an internal concentration polarization (ICP) model. Assumes membrane in AL:FS configuration, therefore
		 * dilutive ICP is encountered.
		 * For more information see Lee et al. 1981 or table 2.1 in "Forward Osmosis Fundamentals and Applications" by Ho Kyong Shon et al.
		@param[in] water_permeability Water permeability coefficient of the membrane (possible units: meter per second per pascal)
		@param[in] pi_draw Osmotic pressure from the draw side (possible units: pascal)
		@param[in] pi_feed Osmotic pressure from the feed side (possible units: pascal)
		@param[in] solute_permeability Permeability of the membrane for solute flux coming from draw to feed (possible units: meter per second)
		@param[in] structural_coef Strutural coefficient of membrane, defines in Equation A10 in "Forward Osmosis Fundamentals and Applications" by Ho Kyong Shon et al. (possible units: meters)
		@param[in] diffusion_coef Diffusion coefficient of solute (possible units: square meters per second)	
		@param[in] tol Tolerance for root finding of Lambert function
		@param[in] max_iterations Maximum iterations in iterative root finding of Lambert function
		@param[in] a_n Initial value in iterative root finding of Lambert function
		\return Optional Water flux in m^3 per m^2 per second, packed in an optional. If Lamberts Function does not find a root, then value is empty
		*/		 		
		export template<std::floating_point T>
		std::optional<T> water_flux_dilutive_icp(T water_permeability, T pi_draw, T pi_feed, T solute_permeability, T structural_coef, T diffusion_coef, T tol=1e-100, int max_iterations=100, T a_n=1.0){
				T D=water_permeability*pi_feed+solute_permeability;
				T E=1.0;
				T res;
			//Use Halley's Method if there is no zero in the K_M denominator		
			if (structural_coef != 0.0){
				T K_m=diffusion_coef/structural_coef;				
				T A=K_m*std::log(water_permeability*pi_draw+solute_permeability);
				T B=-1.0;
				T C=-K_m;
			
				T F=(B/(C*E))*std::exp((B*D-A)/C);
				auto lambert_value=desal::math::lambert(F,tol,max_iterations,a_n);
				
				if (lambert_value){
					res=(C/B)* (*lambert_value)-D;
				}
				else{
					return {};
				}					
							
	//			std::cout<<"Nullstellenauswertung:"<<A+B*res+C*std::log(D+res)<<"\n";
			}
			//Without the linear term (B=0.0) we can solve for the root explicitly
			else{
				T A=diffusion_coef*std::log(water_permeability*pi_draw+solute_permeability);
				T C=-diffusion_coef;			
				res=(std::exp(-A/C)-D)/E;
			}
			return {res};

		}
		
	}
	
}
