@ECHO OFF
cls

set "USE_CUDA=1"

set curpath=%~dp0
set cudapath="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.0\lib\x64\"

echo Building executable
g++ -c %curpath:~0,-1%/../../src/base/math.cpp -pthread -O3 -std=c++20 -fmodules-ts -march=native 
g++ -c %curpath:~0,-1%/../../src/forward_osmosis/water_flux_models.cpp -pthread -O3 -std=c++20 -fmodules-ts -march=native 
g++ main.cpp water_flux_models.o math.o -pthread -O3 -std=c++20 -fmodules-ts -march=native 

del *.o
rmdir /q /s gcm.cache