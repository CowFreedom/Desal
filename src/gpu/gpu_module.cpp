module;

#ifdef opt_use_cuda
	#include "cuda_bindings.h"
#endif


export module desal.gpu;

namespace desal{
	namespace gpu{
		
		/*CUDA: In CUDA the device pointers A_d and B_d refer to allocated memory using cudaMallocPitch2D. Stride therefore refers to pitch in bytes in the cuda memory model*/
		
		
	}
	
}