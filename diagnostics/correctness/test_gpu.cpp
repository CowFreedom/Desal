module;
#include <ostream>

#ifdef opt_use_cuda
	#include "gpu/cuda/test_reductions.h"
#endif

export module tests.correctness:gpu;

import tests;

namespace desal{
	namespace test{

		namespace corr{		
			
			#ifdef opt_use_cuda
			//Tests of gaussnewton cpu
			//CHANGE
			export bool reduce_sum_f32_device_ascending(std::ostream& os, CorrectnessTest& v){
				int reps=10;
				int array_starting_length=5;
				char error_message[200];
				
				v.test_successful=test_reduce_sum_f32_device_ascending(array_starting_length, reps, error_message);
				
				return v.test_successful;
			}
			
			export bool reduce_sum_f32_device_descending(std::ostream& os, CorrectnessTest& v){
				int reps=10;
				int array_starting_length=5;
				char error_message[200];
				
				v.test_successful=test_reduce_sum_f32_device_descending(array_starting_length, reps, error_message);
				
				return v.test_successful;
			}
			
			export bool reduce_sum_f64_device_ascending(std::ostream& os, CorrectnessTest& v){
				int reps=10;
				int array_starting_length=5;
				char error_message[200];
				
				v.test_successful=test_reduce_sum_f64_device_ascending(array_starting_length, reps, error_message);
				
				return v.test_successful;
			}
			
			export bool reduce_sum_f64_device_descending(std::ostream& os, CorrectnessTest& v){
				int reps=10;
				int array_starting_length=5;
				char error_message[200];
				
				v.test_successful=test_reduce_sum_f64_device_descending(array_starting_length, reps, error_message);
				
				return v.test_successful;
			}
			
			export bool reduce_sum_of_squares_poisson_field_residual_f32_device_ascending(std::ostream& os, CorrectnessTest& v){
				int reps=10;
				int array_starting_length=5;
				char error_message[200];
				
				v.test_successful=test_reduce_sum_of_squares_poisson_field_residual_f32_device_ascending(array_starting_length, reps, error_message);
				
				return v.test_successful;
			}	

			
			export bool reduce_sum_of_squares_poisson_field_residual_f32_device_uniform(std::ostream& os, CorrectnessTest& v){
				int reps=5; //TODO: Change to 10
				int array_starting_length=5;
				char error_message[200];
				
				v.test_successful=test_reduce_sum_of_squares_poisson_field_residual_f32_device_uniform(array_starting_length, reps, error_message);
				
				return v.test_successful;
			}					
			#endif
		}
	}

}
