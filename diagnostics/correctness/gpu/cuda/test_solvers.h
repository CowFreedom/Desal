#pragma once

namespace desal{
	
	namespace cuda{
		bool test_mg_vc_poisson_2D_f32_zero_B(int n, int reps, char* error_message=nullptr);
	}
}