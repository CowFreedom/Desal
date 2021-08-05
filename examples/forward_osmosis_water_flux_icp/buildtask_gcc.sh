clear;
g++ -c $PWD/../../src/base/math.cpp -pthread -O3 -std=c++20 -fmodules-ts -march=native 
g++ -c $PWD/../../src/forward_osmosis/water_flux_models.cpp -pthread -O3 -std=c++20 -fmodules-ts -march=native 
g++ -o main.bin main.cpp water_flux_models.o math.o -pthread -O3 -std=c++20 -fmodules-ts -march=native 


rm *.o 
rm -r gcm.cache
./main.bin