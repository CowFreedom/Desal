module;
//#include <iostream>
#include <ostream> //This is already included in the partition but without this additional include compilation it wont work
export module tests.correctness;

//Import GPU code
#ifdef use_gpu
	export import :gpu;
#endif



