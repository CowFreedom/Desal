@ECHO OFF
cls

set "USE_CUDA=1"
set DEBUG_MODE=debug
set curpath=%~dp0
set cudaPath="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.0\lib\x64\"
set windowsSDKPath="C:\Program Files (x86)\Windows Kits\10\Include\10.0.19041.0\um"
set latestDirectX12Headers="%curpath:~0,-1%\include\directx"
echo Building executable in %DEBUG_MODE%


cl /D%DEBUG_MODE% /EHsc /std:c++latest /O2  /c %curpath:~0,-1%\miniengine/renderer.cpp /interface /I %latestDirectX12Headers% /I %windowsSDKPath%
cl /D%DEBUG_MODE% /EHsc /std:c++latest /O2  /c %curpath:~0,-1%\miniengine/miniengine.cpp /interface

cl /D%DEBUG_MODE% /EHsc /std:c++latest /O2 main.cpp miniengine.obj renderer.obj d3d12.lib dxgi.lib dxguid.lib Shell32.lib user32.lib /I %latestDirectX12Headers% /I %windowsSDKPath%

del *.o *.out *.obj *.exp *.lib *.ifc
rmdir /q /s gcm.cache

main.exe