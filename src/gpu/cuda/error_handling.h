#pragma once

#define gpuErrorCheck(ans, os) gpuAssert((ans), __FUNCTION__,__FILE__, __LINE__, os)

namespace desal{
	
	namespace cuda{

		enum class DesalStatus{
			Success,
			CUDAError,
		};

		template<class S>
		inline cudaError_t gpuAssert(cudaError_t code, const char* function, const char* file, int line,S* os){
		#if debug
			if (os){
				if (code != cudaSuccess){
					(*os)<<cudaGetErrorString(code)<<" in "<<function<<" in "<<file<<" in line: "<<line<<"\n";
				}			
			}
		#endif
			return code;
			
		}	
	}

}