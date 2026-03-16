## Build on Windows with GCC and Ninja

```powershell
$env:Path += ";C:\Program Files\CMake\bin"
$ninjaExe = Get-ChildItem "$env:LOCALAPPDATA\Microsoft\WinGet\Packages" -Recurse -Filter ninja.exe -ErrorAction SilentlyContinue | Select-Object -First 1 -ExpandProperty FullName
$env:Path += ";" + (Split-Path $ninjaExe)

cmake -S . -B build -G Ninja -DCMAKE_C_COMPILER="C:/ProgramData/mingw64/mingw64/bin/gcc.exe"
cmake --build build
ctest --test-dir build --output-on-failure