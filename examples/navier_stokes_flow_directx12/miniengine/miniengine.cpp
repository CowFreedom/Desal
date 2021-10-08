module;

#include <Windows.h>
#include <shellapi.h>
//Windows Runtime Library.
#include<wrl.h> 
export module miniengine;

import miniengine.renderer;

namespace miniengine{
	

	export class MiniEngine{
		
		private:
		
		Renderer<3> renderer;
		
		void ParseCommandLineArguments(wchar_t** argv, int argc){
			
			for (size_t i=0; i<argc;i++){
				if (::wcscmp(argv[i],L"-w") == 0 || ::wcscmp(argv[i],L"-w") == 0){
					renderer.g_ClientWidth=::wcstol(argv[++i],nullptr,10);
				}
				if (::wcscmp(argv[i],L"-h") == 0 || ::wcscmp(argv[i],L"-h") == 0){
					renderer.g_ClientHeight=::wcstol(argv[++i],nullptr,10);
				}
			}
			
		}

	
					
				
		public:
		
		MiniEngine(HINSTANCE hInstance, wchar_t** argv, int argc): renderer(Renderer<3>(hInstance)){
			SetThreadDpiAwarenessContext(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2);
			ParseCommandLineArguments(argv,argc);
			
			Renderer<3>::run(&renderer,hInstance);
			
		//	renderer.EnableDebugLayer();
			
		}
		
		
	};
}