#define WIN32_LEAN_AND_MEAN //excludes some less frequently used Windows headers

#include <Windows.h>

#include <shellapi.h>

#if defined(min)
	#undef min
#endif

#if defined(max)
	#undef max
#endif

#if defined(CreateWindow)
	#undef CreateWindow
#endif

//Windows Runtime Library.
#include<wrl.h> 

// Direct 12 specific header files

#include <d3d12.h>

#include <dxgi1_6.h>

#include <d3dcompiler.h>

//#include <DirectXMath.h>

//D3D12 extension

#include <d3dx12.h>

int main(){
	
}