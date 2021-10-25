#undef UNICODE
#define UNICODE
#include <windows.h>

#include <cassert>
#include <algorithm>

#include <iostream>


import miniengine;

class Test{
	public:
	bool g_IsInitialized=false;
};
// see https://devblogs.microsoft.com/oldnewthing/20191014-00/?p=102992

LRESULT CALLBACK WndProc(HWND hwnd, UINT message, WPARAM wParam, LPARAM lParam)	{
			
		//Test* test = reinterpret_cast<Test*>(GetWindowLongPtr(hwnd, GWLP_USERDATA));
	//	LPCREATESTRUCT* cs=(LPCREATESTRUCTW*) lParam;
		
	//	Test* test=(Test*)cs->lpCreateParams;
	 
	/*	
	MessageBox(0,L"Hey",L"N2a",MB_SETFOREGROUND);
	if (test!=nullptr){
		MessageBox(0,L"Drin",L"Drin",MB_SETFOREGROUND);
		
	
	return 0;
	*/
	
	Test* test;
	
	if (message == WM_NCCREATE) {
		LPCREATESTRUCT lpcs = reinterpret_cast<LPCREATESTRUCT>(lParam);
		test = static_cast<Test*>(lpcs->lpCreateParams);
		MessageBox(0,L"Create",L"Create",MB_SETFOREGROUND);	
		SetWindowLongPtr(hwnd, GWLP_USERDATA,reinterpret_cast<LONG_PTR>(test));
	  } else {
	   test = reinterpret_cast<Test*>(GetWindowLongPtr(hwnd, GWLP_USERDATA));
	  }
	  if (test) {
		if (!test->g_IsInitialized)
		MessageBox(0,L"Endlich",L"Endlich",MB_SETFOREGROUND);	
	}
	   //return self->WndProc(message, wParam, lParam);
	  
	  return DefWindowProc(hwnd, message, wParam, lParam);
}	

// maximum mumber of lines the output console should have

//End of File


int CALLBACK wWinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, PWSTR lpCmdLine, int nCmdShow){
	int argc;
	wchar_t** argv=::CommandLineToArgvW(::GetCommandLineW(),&argc);
			
	miniengine::MiniEngine engine(hInstance, argv,argc);
//	MessageBox(0,L"Hey",lpCmdLine,MB_SETFOREGROUND);
	::LocalFree(argv);
	
	
	/*
		const wchar_t* windowClassName=L"TestName";
		HINSTANCE hInst=hInstance;
		
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
			
			int screenWidth = ::GetSystemMetrics(SM_CXSCREEN);
			int screenHeight = ::GetSystemMetrics(SM_CYSCREEN);

			RECT windowRect = { 0, 0, static_cast<LONG>(100), static_cast<LONG>(100) };
			::AdjustWindowRect(&windowRect, WS_OVERLAPPEDWINDOW, FALSE);
			int windowWidth = windowRect.right - windowRect.left;
			int windowHeight = windowRect.bottom - windowRect.top;
			// Center the window within the screen. Clamp to 0, 0 for the top-left corner.
			int windowX = std::max<int>(0, (screenWidth - windowWidth) / 2);
			int windowY = std::max<int>(0, (screenHeight - windowHeight) / 2);
			
			Test test;
			
			Test* blah=&test;
			
			
			HWND hWnd = ::CreateWindowExW(
				NULL,
				windowClassName,
				L"Testtitle",
				WS_OVERLAPPEDWINDOW,
				windowX,
				windowY,
				windowWidth,
				windowHeight,
				NULL,
				NULL,
				hInst,
				blah
				
			);
			MessageBox(0,L"Stop",L"Stop",MB_SETFOREGROUND);				
	*/
//MessageBox(0,L"Hey",windowClassName,MB_SETFOREGROUND);
}