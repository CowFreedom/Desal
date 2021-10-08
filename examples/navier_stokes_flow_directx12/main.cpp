#undef UNICODE
#define UNICODE
#include <windows.h>

import miniengine;




int CALLBACK wWinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, PWSTR lpCmdLine, int nCmdShow){
	int argc;
			
	wchar_t** argv=::CommandLineToArgvW(::GetCommandLineW(),&argc);
			
	miniengine::MiniEngine engine(hInstance, argv,argc);
//	MessageBox(0,L"Hey",lpCmdLine,MB_SETFOREGROUND);
	::LocalFree(argv);
	
//MessageBox(0,L"Hey",windowClassName,MB_SETFOREGROUND);
}