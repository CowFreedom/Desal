#pragma once

namespace desal{
	
	namespace cuda{
		bool test_reduce_sum_f32_device_ascending(int n, int reps, char* error_message=nullptr);

		bool test_reduce_sum_f32_device_descending(int n, int reps, char* error_message=nullptr);

		bool test_reduce_sum_f64_device_ascending(int n, int reps, char* error_message=nullptr);
							  
		bool test_reduce_sum_f64_device_descending(int n, int reps, char* error_message=nullptr);

		bool test_reduce_sum_of_squares_poisson_field_residual_f32_device_ascending(int n, int reps, char* error_message=nullptr);

		bool test_reduce_sum_of_squares_poisson_field_residual_f32_device_uniform(int reps, char* error_message=nullptr);

	}
}