@ECHO OFF
cls

set "USE_CUDA=1"

set curpath=%~dp0
set cudapath="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.0\lib\x64\"

echo Building executable
nvcc %curpath:~0,-1%\..\..\src\gpu\solvers\poisson_multigrid.cu -c -o gpu_poisson.obj 
nvcc %curpath:~0,-1%\..\..\src\gpu\hostgpu_bindings.cu -c -o hostgpu_bindings.obj
nvcc %curpath:~0,-1%\..\..\src\gpu\reductions.cu -c -o gpu_reductions.obj
::cl /EHsc /std:c++latest /O2 %curpath:~0,-1%\main.cpp gpu_advection.obj /link /LIBPATH:%cudapath% cudart.lib
nvcc  %curpath:~0,-1%\main.cu gpu_poisson.obj gpu_reductions.obj hostgpu_bindings.obj -o main.exe

::g++ -c %curpath:~0,-1%/../../src/base/math.cpp -pthread -O3 -std=c++20 -fmodules-ts -march=native 
::g++ -c %curpath:~0,-1%/../../src/forward_osmosis/water_flux_models.cpp -pthread -O3 -std=c++20 -fmodules-ts -march=native 
::g++ main.cpp water_flux_models.o math.o -pthread -O3 -std=c++20 -fmodules-ts -march=native 



del *.o *.out *.obj *.exp *.lib
rmdir /q /s gcm.cache

main.exe