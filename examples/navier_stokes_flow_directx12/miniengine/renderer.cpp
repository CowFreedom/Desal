module;

#define WIN32_LEAN_AND_MEAN //excludes some less frequently used Windows headers
#undef UNICODE
#define UNICODE
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

//D3D12 extension. Included in the official GitHub DirectX12 headers but not in the Windows10SDK as of yet
#include <d3dx12.h>
#include <cassert>
#include <chrono>
#include <stdio.h>
export module miniengine.renderer;

using namespace Microsoft::WRL;

namespace miniengine{
	
		

	inline void ThrowIfFailed(HRESULT hr){
		
		if (FAILED(hr)){
			throw std::exception();
			
		}		
	}


	
	export 
	template<int NUMFRAMES>	
	class Renderer{
		private:
		const uint8_t g_NumFrames=NUMFRAMES;
		
	static LRESULT CALLBACK WndProc(HWND hwnd, UINT message, WPARAM wParam, LPARAM lParam)	{
			
		Renderer<NUMFRAMES>* renderer = reinterpret_cast<Renderer<NUMFRAMES>*>(GetWindowLongPtr(hwnd, GWLP_USERDATA));
		//CREATESTRUCT* cs=(CREATESTRUCT*) lParam;
		//Renderer<NUMFRAMES>* renderer=(Renderer<NUMFRAMES>*)cs->lpCreateParams;
	//	MessageBox(0,L"Not start",L"start",MB_SETFOREGROUND);
		//Renderer* renderer;
				
		if (message== WM_CREATE)
		{
			// Save the DXSample* passed in to CreateWindow.
			LPCREATESTRUCT pCreateStruct = reinterpret_cast<LPCREATESTRUCT>(lParam);
			SetWindowLongPtr(hwnd, GWLP_USERDATA, reinterpret_cast<LONG_PTR>(pCreateStruct->lpCreateParams));
		}
		
		if(renderer){
			if (renderer->g_IsInitialized)
			{
				switch (message){
					
				case WM_CREATE:
				{
					// Save the DXSample* passed in to CreateWindow.
					LPCREATESTRUCT pCreateStruct = reinterpret_cast<LPCREATESTRUCT>(lParam);
					SetWindowLongPtr(hwnd, GWLP_USERDATA, reinterpret_cast<LONG_PTR>(pCreateStruct->lpCreateParams));
				}

				case WM_PAINT:
					renderer->Update();
					renderer->Render();
					break;
					
				case WM_SYSKEYDOWN:
				case WM_KEYDOWN:
				{
					bool alt = (::GetAsyncKeyState(VK_MENU) & 0x8000) != 0;

					switch (wParam)
					{
					case 'V':
						renderer->g_VSync = !renderer->g_VSync;
						break;
					case VK_ESCAPE:
						::PostQuitMessage(0);
						break;
					case VK_RETURN:
						if ( alt )
						{
					case VK_F11:
						renderer->SetFullscreen(!renderer->g_Fullscreen);
						}
						break;
					}
				}
				break;
				// The default window procedure will play a system notification sound 
				// when pressing the Alt+Enter keyboard combination if this message is 
				// not handled.
				case WM_SYSCHAR:
				break;		

				case WM_SIZE:
				{
					RECT clientRect = {};
					::GetClientRect(renderer->g_hWnd, &clientRect);

					int width = clientRect.right - clientRect.left;
					int height = clientRect.bottom - clientRect.top;

					renderer->Resize(width, height);
					break;	
				}					
			   case WM_DESTROY:
					::PostQuitMessage(0);
					break;
				default:
					return ::DefWindowProcW(hwnd, message, wParam, lParam);
				}
			}
		}
		else
		{
	//	MessageBox(0,L"Not initialized",L"Notinit",MB_SETFOREGROUND);
				
			return ::DefWindowProcW(hwnd, message, wParam, lParam);
		}
	}	
		

		public:
		Renderer(HINSTANCE hInstance){
			EnableDebugLayer();
			printf("RegisterClassEx error: %d\n",GetLastError());
			g_TearingSupported = CheckTearingSupport();


					
		}
		
	static int run(Renderer* renderer, HINSTANCE hInstance){
			const wchar_t* windowClassName= L"DX12WindowClass";		
			renderer->RegisterWindowClass(hInstance, windowClassName);
			
			renderer->g_hWnd = renderer->CreateWindow(windowClassName, hInstance, L"Learning DirectX 12",renderer->g_ClientWidth, renderer->g_ClientHeight);
	
			// Initialize the global window rect variable.
			::GetWindowRect(renderer->g_hWnd, &(renderer->g_WindowRect));			
					
			ComPtr<IDXGIAdapter4> dxgiAdapter4 = renderer->GetAdapter();

			renderer->g_Device = renderer->CreateDevice(dxgiAdapter4);

			renderer->g_CommandQueue = renderer->CreateCommandQueue(renderer->g_Device, D3D12_COMMAND_LIST_TYPE_DIRECT);

			renderer->g_SwapChain = renderer->CreateSwapChain(renderer->g_hWnd, renderer->g_CommandQueue,renderer->g_ClientWidth, renderer->g_ClientHeight,renderer->g_NumFrames);

			renderer->g_CurrentBackBufferIndex = renderer->g_SwapChain->GetCurrentBackBufferIndex();

			renderer->g_RTVDescriptorHeap = renderer->CreateDescriptorHeap(renderer->g_Device, D3D12_DESCRIPTOR_HEAP_TYPE_RTV, renderer->g_NumFrames);
			renderer->g_RTVDescriptorSize = renderer->g_Device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);

			renderer->UpdateRenderTargetViews(renderer->g_Device, renderer->g_SwapChain, renderer->g_RTVDescriptorHeap);
			
			for (int i = 0; i < renderer->g_NumFrames; ++i)
			{
				renderer->g_CommandAllocators[i] = renderer->CreateCommandAllocator(renderer->g_Device, D3D12_COMMAND_LIST_TYPE_DIRECT);
			}
			renderer->g_CommandList = renderer->CreateCommandList(renderer->g_Device,
			renderer->g_CommandAllocators[renderer->g_CurrentBackBufferIndex], D3D12_COMMAND_LIST_TYPE_DIRECT);		

			renderer->g_Fence = renderer->CreateFence(renderer->g_Device);
			renderer->g_FenceEvent = renderer->CreateEventHandle();			
		
			renderer->g_IsInitialized = true;
			
			::ShowWindow(renderer->g_hWnd, SW_SHOW);

			MSG msg = {};
			while (msg.message != WM_QUIT)
			{
				if (::PeekMessage(&msg, NULL, 0, 0, PM_REMOVE))
				{
					::TranslateMessage(&msg);
					::DispatchMessage(&msg);
				}
			}		
			return 0;
		
		}
		
		~Renderer(){
			    // Make sure the command queue has finished all commands before closing.
			Flush(g_CommandQueue, g_Fence, g_FenceValue, g_FenceEvent);
			::CloseHandle(g_FenceEvent);
					
		}

		
		bool g_IsInitialized=false;
		int g_ClientWidth=1280;
		int g_ClientHeight=720;
		bool g_Fullscreen;
		
		HWND g_hWnd;
		RECT g_WindowRect;
		
		//DirectX12 Objects
		
		ComPtr<ID3D12Device2> g_Device;
		ComPtr<ID3D12CommandQueue> g_CommandQueue;
		ComPtr<IDXGISwapChain4> g_SwapChain;
		ComPtr<ID3D12Resource> g_BackBuffers[NUMFRAMES];
		ComPtr<ID3D12GraphicsCommandList> g_CommandList;
		ComPtr<ID3D12CommandAllocator> g_CommandAllocators[NUMFRAMES];
		ComPtr<ID3D12DescriptorHeap> g_RTVDescriptorHeap;
	
		UINT g_RTVDescriptorSize;
		UINT g_CurrentBackBufferIndex;
		
		//Synchronization Objects
				
		ComPtr<ID3D12Fence> g_Fence;
		uint64_t g_FrameFenceValues[NUMFRAMES]={};
		uint64_t g_FenceValue=0;

		
		HANDLE g_FenceEvent;
		
		bool g_VSync=true; //can be toggled with V key
		
		bool g_TearingSupported=false;//Can be toggled with F11

		//This struct holds information for the Window class (like a resource description)
		struct tagWNDCLASSEXW {
			UINT        cbSize;
			UINT        style;
			WNDPROC     lpfnWndProc;
			int         cbClsExtra;
			int         cbWndExtra;
			HINSTANCE   hInstance;
			HICON       hIcon;
			HCURSOR     hCursor;
			HBRUSH      hbrBackground;
			LPCWSTR     lpszMenuName;
			LPCWSTR     lpszClassName;
			HICON       hIconSm;
		};

		using WINDCLASSEXW=tagWNDCLASSEXW;
		using PWNDCLASSEXW=WINDCLASSEXW*;	
	
	
		void EnableDebugLayer(){
			#if defined (debug)
			ComPtr<ID3D12Debug> debugInterface;
			ThrowIfFailed(D3D12GetDebugInterface(IID_PPV_ARGS(&debugInterface)));
			
			debugInterface->EnableDebugLayer();
			#endif
		}
		
		void RegisterWindowClass(HINSTANCE hInst, const wchar_t* windowClassName ){			
			
			// Register a window class for creating our render window with.
			WNDCLASSEXW windowClass = {};
			windowClass.cbSize = sizeof(WNDCLASSEX);
			windowClass.style = CS_HREDRAW | CS_VREDRAW;
			windowClass.lpfnWndProc = &WndProc;
			windowClass.cbClsExtra = 0;
			windowClass.cbWndExtra = 0;
			windowClass.hInstance = hInst;
			windowClass.hIcon = ::LoadIcon(hInst, NULL);
			windowClass.hCursor = ::LoadCursor(NULL, IDC_ARROW);
			windowClass.hbrBackground = (HBRUSH)(COLOR_WINDOW + 1);
			windowClass.lpszMenuName = NULL;
			windowClass.lpszClassName = windowClassName;
			windowClass.hIconSm = ::LoadIcon(hInst, NULL);

			static HRESULT hr = ::RegisterClassExW(&windowClass);
			assert(SUCCEEDED(hr));
		//	assert(atom > 0);
		}
		
		//Creates a window with the given classname
		HWND CreateWindow(const wchar_t* windowClassName, HINSTANCE hInst, const wchar_t* windowTitle, uint32_t width, uint32_t height)
		{

			int screenWidth = ::GetSystemMetrics(SM_CXSCREEN);
			int screenHeight = ::GetSystemMetrics(SM_CYSCREEN);

			RECT windowRect = { 0, 0, static_cast<LONG>(width), static_cast<LONG>(height) };
			::AdjustWindowRect(&windowRect, WS_OVERLAPPEDWINDOW, FALSE);
			int windowWidth = windowRect.right - windowRect.left;
			int windowHeight = windowRect.bottom - windowRect.top;
			// Center the window within the screen. Clamp to 0, 0 for the top-left corner.
			int windowX = std::max<int>(0, (screenWidth - windowWidth) / 2);
			int windowY = std::max<int>(0, (screenHeight - windowHeight) / 2);
			
			HWND hWnd = ::CreateWindowExW(
				NULL,
				windowClassName,
				windowTitle,
				WS_OVERLAPPEDWINDOW,
				windowX,
				windowY,
				windowWidth,
				windowHeight,
				NULL,
				NULL,
				hInst,
				this
			);
			//MessageBox(0,L"Stop",L"Stop",MB_SETFOREGROUND);		
			assert(hWnd && "Failed to create window");
			
			return hWnd;
		}
		
		
		ComPtr<IDXGIAdapter4> GetAdapter()
		{
			ComPtr<IDXGIFactory4> dxgiFactory;
			UINT createFactoryFlags = 0;
			
			#if defined(debug)
			createFactoryFlags = DXGI_CREATE_FACTORY_DEBUG;
			#endif

			ThrowIfFailed(CreateDXGIFactory2(createFactoryFlags, IID_PPV_ARGS(&dxgiFactory)));
			ComPtr<IDXGIAdapter1> dxgiAdapter1;
			ComPtr<IDXGIAdapter4> dxgiAdapter4;			
			SIZE_T maxDedicatedVideoMemory = 0;
			
			for (UINT i = 0; dxgiFactory->EnumAdapters1(i, &dxgiAdapter1) != DXGI_ERROR_NOT_FOUND; ++i)
			{
				DXGI_ADAPTER_DESC1 dxgiAdapterDesc1;
				dxgiAdapter1->GetDesc1(&dxgiAdapterDesc1);

				// Check to see if the adapter can create a D3D12 device without actually 
				// creating it. The adapter with the largest dedicated video memory
				// is favored.
				if ((dxgiAdapterDesc1.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) == 0 &&
					SUCCEEDED(D3D12CreateDevice(dxgiAdapter1.Get(), 
						D3D_FEATURE_LEVEL_12_0, __uuidof(ID3D12Device), nullptr)) && 
					dxgiAdapterDesc1.DedicatedVideoMemory > maxDedicatedVideoMemory )
				{
					maxDedicatedVideoMemory = dxgiAdapterDesc1.DedicatedVideoMemory;
					
				//	MessageBox(0,L"Not initialized",std::to_wstring(maxDedicatedVideoMemory).c_str(),MB_SETFOREGROUND);
					
					ThrowIfFailed(dxgiAdapter1.As(&dxgiAdapter4));
				}
			}
			return dxgiAdapter4;		
		}
		
		ComPtr<ID3D12Device2> CreateDevice(ComPtr<IDXGIAdapter4> adapter)
		{
			ComPtr<ID3D12Device2> d3d12Device2;
			ThrowIfFailed(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&d3d12Device2)));
			// Enable debug messages in debug mode.
		#if defined(_DEBUG)
			ComPtr<ID3D12InfoQueue> pInfoQueue;
			if (SUCCEEDED(d3d12Device2.As(&pInfoQueue)))
			{
				pInfoQueue->SetBreakOnSeverity(D3D12_MESSAGE_SEVERITY_CORRUPTION, TRUE);
				pInfoQueue->SetBreakOnSeverity(D3D12_MESSAGE_SEVERITY_ERROR, TRUE);
				pInfoQueue->SetBreakOnSeverity(D3D12_MESSAGE_SEVERITY_WARNING, TRUE);
				// Suppress whole categories of messages
				//D3D12_MESSAGE_CATEGORY Categories[] = {};

				// Suppress messages based on their severity level
				D3D12_MESSAGE_SEVERITY Severities[] =
				{
					D3D12_MESSAGE_SEVERITY_INFO
				};

				// Suppress individual messages by their ID
				D3D12_MESSAGE_ID DenyIds[] = {
					D3D12_MESSAGE_ID_CLEARRENDERTARGETVIEW_MISMATCHINGCLEARVALUE,   // I'm really not sure how to avoid this message.
					D3D12_MESSAGE_ID_MAP_INVALID_NULLRANGE,                         // This warning occurs when using capture frame while graphics debugging.
					D3D12_MESSAGE_ID_UNMAP_INVALID_NULLRANGE,                       // This warning occurs when using capture frame while graphics debugging.
				};

				D3D12_INFO_QUEUE_FILTER NewFilter = {};
				//NewFilter.DenyList.NumCategories = _countof(Categories);
				//NewFilter.DenyList.pCategoryList = Categories;
				NewFilter.DenyList.NumSeverities = _countof(Severities);
				NewFilter.DenyList.pSeverityList = Severities;
				NewFilter.DenyList.NumIDs = _countof(DenyIds);
				NewFilter.DenyList.pIDList = DenyIds;

				ThrowIfFailed(pInfoQueue->PushStorageFilter(&NewFilter));
			}
		#endif

			return d3d12Device2;
		}
		
		ComPtr<ID3D12CommandQueue> CreateCommandQueue(ComPtr<ID3D12Device2> device, D3D12_COMMAND_LIST_TYPE type )
		{
			ComPtr<ID3D12CommandQueue> d3d12CommandQueue;

			D3D12_COMMAND_QUEUE_DESC desc = {};
			desc.Type =     type;
			desc.Priority = D3D12_COMMAND_QUEUE_PRIORITY_NORMAL;
			desc.Flags =    D3D12_COMMAND_QUEUE_FLAG_NONE;
			desc.NodeMask = 0;

			ThrowIfFailed(device->CreateCommandQueue(&desc, IID_PPV_ARGS(&d3d12CommandQueue)));

			return d3d12CommandQueue;
		}
		
		bool CheckTearingSupport()
		{
			BOOL allowTearing = FALSE;

			// Rather than create the DXGI 1.5 factory interface directly, we create the
			// DXGI 1.4 interface and query for the 1.5 interface. This is to enable the 
			// graphics debugging tools which will not support the 1.5 factory interface 
			// until a future update.
			ComPtr<IDXGIFactory4> factory4;
			if (SUCCEEDED(CreateDXGIFactory1(IID_PPV_ARGS(&factory4))))
			{
				ComPtr<IDXGIFactory5> factory5;
				if (SUCCEEDED(factory4.As(&factory5)))
				{
					if (FAILED(factory5->CheckFeatureSupport(
						DXGI_FEATURE_PRESENT_ALLOW_TEARING, 
						&allowTearing, sizeof(allowTearing))))
					{
						allowTearing = FALSE;
					}
				}
			}

			return allowTearing == TRUE;
		}
		
		ComPtr<IDXGISwapChain4> CreateSwapChain(HWND hWnd, 
			ComPtr<ID3D12CommandQueue> commandQueue, 
			uint32_t width, uint32_t height, uint32_t bufferCount )
		{
			ComPtr<IDXGISwapChain4> dxgiSwapChain4;
			ComPtr<IDXGIFactory4> dxgiFactory4;
			UINT createFactoryFlags = 0;
		#if defined(_DEBUG)
			createFactoryFlags = DXGI_CREATE_FACTORY_DEBUG;
		#endif

			ThrowIfFailed(CreateDXGIFactory2(createFactoryFlags, IID_PPV_ARGS(&dxgiFactory4)));		
			
			DXGI_SWAP_CHAIN_DESC1 swapChainDesc = {};
			swapChainDesc.Width = width;
			swapChainDesc.Height = height;
			swapChainDesc.Format = DXGI_FORMAT_R8G8B8A8_UNORM;
			swapChainDesc.Stereo = FALSE;
			swapChainDesc.SampleDesc = { 1, 0 };
			swapChainDesc.BufferUsage = DXGI_USAGE_RENDER_TARGET_OUTPUT;
			swapChainDesc.BufferCount = bufferCount;
			swapChainDesc.Scaling = DXGI_SCALING_STRETCH;
			swapChainDesc.SwapEffect = DXGI_SWAP_EFFECT_FLIP_DISCARD;
			swapChainDesc.AlphaMode = DXGI_ALPHA_MODE_UNSPECIFIED;
			// It is recommended to always allow tearing if tearing support is available.
			swapChainDesc.Flags = CheckTearingSupport() ? DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING : 0;	

			ComPtr<IDXGISwapChain1> swapChain1;
			ThrowIfFailed(dxgiFactory4->CreateSwapChainForHwnd(
				commandQueue.Get(),
				hWnd,
				&swapChainDesc,
				nullptr,
				nullptr,
				&swapChain1));

			// Disable the Alt+Enter fullscreen toggle feature. Switching to fullscreen
			// will be handled manually.
			ThrowIfFailed(dxgiFactory4->MakeWindowAssociation(hWnd, DXGI_MWA_NO_ALT_ENTER));

			ThrowIfFailed(swapChain1.As(&dxgiSwapChain4));

			return dxgiSwapChain4;
		}		

		ComPtr<ID3D12DescriptorHeap> CreateDescriptorHeap(ComPtr<ID3D12Device2> device,
			D3D12_DESCRIPTOR_HEAP_TYPE type, uint32_t numDescriptors)
		{
			ComPtr<ID3D12DescriptorHeap> descriptorHeap;
			D3D12_DESCRIPTOR_HEAP_DESC desc = {};

			desc.NumDescriptors = numDescriptors;
			desc.Type = type;
			ThrowIfFailed(device->CreateDescriptorHeap(&desc, IID_PPV_ARGS(&descriptorHeap)));

			return descriptorHeap;
		}	
		
		void UpdateRenderTargetViews(ComPtr<ID3D12Device2> device, ComPtr<IDXGISwapChain4> swapChain, ComPtr<ID3D12DescriptorHeap> descriptorHeap)	{
			auto rtvDescriptorSize = device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_RTV);
			CD3DX12_CPU_DESCRIPTOR_HANDLE rtvHandle(descriptorHeap->GetCPUDescriptorHandleForHeapStart());

			for (int i = 0; i < g_NumFrames; ++i)    {
				ComPtr<ID3D12Resource> backBuffer;
				ThrowIfFailed(swapChain->GetBuffer(i, IID_PPV_ARGS(&backBuffer)));
				device->CreateRenderTargetView(backBuffer.Get(), nullptr, rtvHandle);
				g_BackBuffers[i] = backBuffer;
				rtvHandle.Offset(rtvDescriptorSize);
			}

		}
		
		ComPtr<ID3D12CommandAllocator> CreateCommandAllocator(ComPtr<ID3D12Device2> device,	D3D12_COMMAND_LIST_TYPE type)
		{
			ComPtr<ID3D12CommandAllocator> commandAllocator;
			ThrowIfFailed(device->CreateCommandAllocator(type, IID_PPV_ARGS(&commandAllocator)));
			return commandAllocator;
		}

		ComPtr<ID3D12GraphicsCommandList> CreateCommandList(ComPtr<ID3D12Device2> device,
			ComPtr<ID3D12CommandAllocator> commandAllocator, D3D12_COMMAND_LIST_TYPE type)
		{
			ComPtr<ID3D12GraphicsCommandList> commandList;
			ThrowIfFailed(device->CreateCommandList(0, type, commandAllocator.Get(), nullptr, IID_PPV_ARGS(&commandList)));
			ThrowIfFailed(commandList->Close());
			return commandList;
		}
		
		ComPtr<ID3D12Fence> CreateFence(ComPtr<ID3D12Device2> device)
		{
			ComPtr<ID3D12Fence> fence;

			ThrowIfFailed(device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence)));

			return fence;
		}

		HANDLE CreateEventHandle(){
			HANDLE fenceEvent;

			fenceEvent = ::CreateEvent(NULL, FALSE, FALSE, NULL);
			assert(fenceEvent && "Failed to create fence event.");
			return fenceEvent;
		}
		
		uint64_t Signal(ComPtr<ID3D12CommandQueue> commandQueue, ComPtr<ID3D12Fence> fence,	uint64_t& fenceValue){
			uint64_t fenceValueForSignal = ++fenceValue;
			ThrowIfFailed(commandQueue->Signal(fence.Get(), fenceValueForSignal));
			return fenceValueForSignal;
			
		}
		
		void WaitForFenceValue(ComPtr<ID3D12Fence> fence, uint64_t fenceValue, HANDLE fenceEvent,std::chrono::milliseconds duration = std::chrono::milliseconds::max() )
		{
			if (fence->GetCompletedValue() < fenceValue)
			{
				ThrowIfFailed(fence->SetEventOnCompletion(fenceValue, fenceEvent));
				::WaitForSingleObject(fenceEvent, static_cast<DWORD>(duration.count()));
			}
		}		
		
		void Flush(ComPtr<ID3D12CommandQueue> commandQueue, ComPtr<ID3D12Fence> fence,
			uint64_t& fenceValue, HANDLE fenceEvent )
		{
			uint64_t fenceValueForSignal = Signal(commandQueue, fence, fenceValue);
			WaitForFenceValue(fence, fenceValueForSignal, fenceEvent);
		}		
		
		void Update(){

			static uint64_t frameCounter = 0;
			static double elapsedSeconds = 0.0;
			static std::chrono::high_resolution_clock clock;

			static auto t0 = clock.now();

			frameCounter++;

			auto t1 = clock.now();

			auto deltaTime = t1 - t0;
			t0 = t1;
			
			elapsedSeconds += deltaTime.count() * 1e-9;


			if (elapsedSeconds > 1.0)
	
	
			{
				char buffer[500];
		
				auto fps = frameCounter / elapsedSeconds;
		
				sprintf_s(buffer, 500, "FPS: %f\n", fps);
				std::wstring wc( 500, L'#' );
				mbstowcs( &wc[0], buffer, 500 );
		//MultiByteToWideChar(CP_UTF8, 0, &str[0], (int)str.size(), &wstrTo[0], size_needed);
				OutputDebugString(wc.c_str()); //Todo
				frameCounter = 0;
				elapsedSeconds = 0.0;
			}

		}
				

		void Render(){
			auto commandAllocator = g_CommandAllocators[g_CurrentBackBufferIndex];
			auto backBuffer = g_BackBuffers[g_CurrentBackBufferIndex];

			commandAllocator->Reset();
			g_CommandList->Reset(commandAllocator.Get(), nullptr);
								
			// Clear the render target.
			{
				CD3DX12_RESOURCE_BARRIER barrier = CD3DX12_RESOURCE_BARRIER::Transition(backBuffer.Get(),D3D12_RESOURCE_STATE_PRESENT, D3D12_RESOURCE_STATE_RENDER_TARGET);

				g_CommandList->ResourceBarrier(1, &barrier);
				//FLOAT clearColor[] = { 0.4f, 0.6f, 0.9f, 1.0f };
				FLOAT clearColor[] = { 0.4f, 0.6f, 0.9f, 1.0f };

				CD3DX12_CPU_DESCRIPTOR_HANDLE rtv(g_RTVDescriptorHeap->GetCPUDescriptorHandleForHeapStart(),g_CurrentBackBufferIndex, g_RTVDescriptorSize);

				g_CommandList->ClearRenderTargetView(rtv, clearColor, 0, nullptr);


			}
			// Present
			{
				CD3DX12_RESOURCE_BARRIER barrier = CD3DX12_RESOURCE_BARRIER::Transition(backBuffer.Get(),D3D12_RESOURCE_STATE_RENDER_TARGET, D3D12_RESOURCE_STATE_PRESENT);
				g_CommandList->ResourceBarrier(1, &barrier);
				ThrowIfFailed(g_CommandList->Close());

				ID3D12CommandList* const commandLists[] = {g_CommandList.Get()};
				g_CommandQueue->ExecuteCommandLists(_countof(commandLists), commandLists);	

				UINT syncInterval = g_VSync ? 1 : 0;
				UINT presentFlags = g_TearingSupported && !g_VSync ? DXGI_PRESENT_ALLOW_TEARING : 0;

				ThrowIfFailed(g_SwapChain->Present(syncInterval, presentFlags));
				g_FrameFenceValues[g_CurrentBackBufferIndex] = Signal(g_CommandQueue, g_Fence, g_FenceValue);	

				g_CurrentBackBufferIndex = g_SwapChain->GetCurrentBackBufferIndex();

				WaitForFenceValue(g_Fence, g_FrameFenceValues[g_CurrentBackBufferIndex], g_FenceEvent);

			}
		}

		void Resize(uint32_t width, uint32_t height){
			if (g_ClientWidth != width || g_ClientHeight != height){
				// Don't allow 0 size swap chain back buffers.
				g_ClientWidth = std::max(1u, width );
				g_ClientHeight = std::max( 1u, height);

				// Flush the GPU queue to make sure the swap chain's back buffers
				// are not being referenced by an in-flight command list.
				Flush(g_CommandQueue, g_Fence, g_FenceValue, g_FenceEvent);
				
				for (int i = 0; i < g_NumFrames; ++i){

					// Any references to the back buffers must be released
					// before the swap chain can be resized.
					g_BackBuffers[i].Reset();
					g_FrameFenceValues[i] = g_FrameFenceValues[g_CurrentBackBufferIndex];
				}
				
				DXGI_SWAP_CHAIN_DESC swapChainDesc = {};
				ThrowIfFailed(g_SwapChain->GetDesc(&swapChainDesc));
				ThrowIfFailed(g_SwapChain->ResizeBuffers(g_NumFrames, g_ClientWidth, g_ClientHeight,swapChainDesc.BufferDesc.Format, swapChainDesc.Flags));

				g_CurrentBackBufferIndex = g_SwapChain->GetCurrentBackBufferIndex();

				UpdateRenderTargetViews(g_Device, g_SwapChain, g_RTVDescriptorHeap);
			}
		}				
		
		void SetFullscreen(bool fullscreen)
		{
			if (g_Fullscreen != fullscreen)
			{
				g_Fullscreen = fullscreen;

				if (g_Fullscreen) // Switching to fullscreen.
				{
					// Store the current window dimensions so they can be restored 
					// when switching out of fullscreen state.
					::GetWindowRect(g_hWnd, &g_WindowRect);		
					// Set the window style to a borderless window so the client area fills
					// the entire screen.
					UINT windowStyle = WS_OVERLAPPEDWINDOW & ~(WS_CAPTION | WS_SYSMENU | WS_THICKFRAME | WS_MINIMIZEBOX | WS_MAXIMIZEBOX);

					::SetWindowLongW(g_hWnd, GWL_STYLE, windowStyle);
					
					// Query the name of the nearest display device for the window.
					// This is required to set the fullscreen dimensions of the window
					// when using a multi-monitor setup.
					HMONITOR hMonitor = ::MonitorFromWindow(g_hWnd, MONITOR_DEFAULTTONEAREST);
					MONITORINFOEX monitorInfo = {};
					monitorInfo.cbSize = sizeof(MONITORINFOEX);
					::GetMonitorInfo(hMonitor, &monitorInfo);	
					
					::SetWindowPos(g_hWnd, HWND_TOP,
						monitorInfo.rcMonitor.left,
						monitorInfo.rcMonitor.top,
						monitorInfo.rcMonitor.right - monitorInfo.rcMonitor.left,
						monitorInfo.rcMonitor.bottom - monitorInfo.rcMonitor.top,
						SWP_FRAMECHANGED | SWP_NOACTIVATE);

					::ShowWindow(g_hWnd, SW_MAXIMIZE);
				}		
				else
				{
					// Restore all the window decorators.
					::SetWindowLong(g_hWnd, GWL_STYLE, WS_OVERLAPPEDWINDOW);

					::SetWindowPos(g_hWnd, HWND_NOTOPMOST,
						g_WindowRect.left,
						g_WindowRect.top,
						g_WindowRect.right - g_WindowRect.left,
						g_WindowRect.bottom - g_WindowRect.top,
						SWP_FRAMECHANGED | SWP_NOACTIVATE);

					::ShowWindow(g_hWnd, SW_NORMAL);
				}
			}
		}				
			
	};
	
		

	
}