@ECHO OFF
cls

set "USE_CUDA=1"
set DEBUG_MODE=debug
set curpath=%~dp0
set cudapath="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.0\lib\x64\"

echo Building executable in %DEBUG_MODE%

if defined USE_CUDA (
	::Building CUDA sources
	nvcc -std=c++17 -O3 %curpath:~0,-1%\..\..\src\gpu\cuda\solvers\advection.cu -c -o gpu_advection.obj
	
	nvcc -std=c++17 -O3 %curpath:~0,-1%\..\..\diagnostics\correctness\gpu\cuda\utility.cu -c -o gpu_test_utility.obj
	

	nvcc -std=c++17 -D %DEBUG_MODE% -O3  %curpath:~0,-1%\main.cu gpu_advection.obj gpu_test_utility.obj -o main.exe
)

del *.o *.out *.obj *.exp *.lib *.ifc
rmdir /q /s gcm.cache

main.exe