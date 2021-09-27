@ECHO OFF
cls

set "USE_CUDA=1"

set curpath=%~dp0
set cudapath="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.0\lib\x64\"

echo Building executable

if defined USE_CUDA (
	::Building CUDA sources
	nvcc %curpath:~0,-1%\..\src\gpu\cuda\solvers\poisson_multigrid.cu -c -o gpu_poisson.obj
	nvcc %curpath:~0,-1%\..\src\gpu\cuda\reductions.cu -c -o gpu_reductions.obj
	nvcc %curpath:~0,-1%\..\src\gpu\cuda\transformations.cu -c -o gpu_transformations.obj
	
	::Building GPU Module
	cl /Duse_cuda /EHsc /std:c++latest /O2  /c %curpath:~0,-1%\..\src\gpu\gpu_module.cpp /interface
	
	::Building correctness test modules
	cl /EHsc /std:c++latest /O2  /c %curpath:~0,-1%\tests.cpp /interface
	
	::Building CUDA GPU tests
	nvcc %curpath:~0,-1%\correctness\gpu\cuda\test_reductions.cu -c -o gpu_test_reductions.obj
	nvcc %curpath:~0,-1%\correctness\gpu\cuda\test_solvers.cu -c -o gpu_test_solvers.obj
	
	::Building gpu correctness test submodules and entire module
	cl /Duse_cuda /EHsc /std:c++latest /O2  /c %curpath:~0,-1%\correctness/test_gpu.cpp /interface	
	cl /Duse_cuda /EHsc /std:c++latest /O2  /c %curpath:~0,-1%\correctness/correctness.cpp /interface
	
	nvcc %curpath:~0,-1%\correctness\gpu\cuda\utility.cu -c -o gpu_test_utility.obj
	
	cl /EHsc /Duse_cuda /std:c++latest /O2  %curpath:~0,-1%\run_diagnostics.cpp test_gpu.obj gpu_test_solvers.obj gpu_transformations.obj gpu_test_reductions.obj gpu_test_utility.obj gpu_module.obj tests.obj correctness.obj gpu_poisson.obj gpu_reductions.obj /link /LIBPATH:%cudapath% cudart.lib

	
)





del *.o *.out *.obj *.exp *.lib *.ifc
rmdir /q /s gcm.cache

run_diagnostics.exe