@echo off
setlocal
setlocal enabledelayedexpansion
set /a cores=%number_of_processors%-2 > null

set build_platform=x64

set source_path=%~dp0

set build_directory=%source_path%\build

if EXIST "%build_directory%" (
    call:rm_rebuild_dir "%build_directory%"
) 

md "%build_directory%"
set vcvarsall_dir=D:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\vcvarsall.bat

IF NOT EXIST "%vcvarsall_dir%" (
  goto set_vcvarsall_dir
) else (
  goto cmake
)

:set_vcvarsall_dir
SET /P vcvarsall_dir="Please input the path of visual studio command Prompt, such as C:\Program Files (x86)\Microsoft Visual Studio 14.0\VC\vcvarsall.bat   =======>"
set tmp_var=!vcvarsall_dir!
call:remove_space
set vcvarsall_dir=!tmp_var!   
IF NOT EXIST "!vcvarsall_dir!" (
    echo "------------!vcvarsall_dir! not exist------------"
    goto set_vcvarsall_dir
)

:cmake
%~d0
cd "%build_directory%"

cmake ..  -G "Visual Studio 14 2015" -A %build_platform%

if "%build_platform%"=="x64" (
    call "%vcvarsall_dir%" amd64
) else (
    call "%vcvarsall_dir%" x86   
)

msbuild /maxcpucount:%cores% /p:Configuration=Release main.vcxproj

goto:eof

:rm_rebuild_dir
    del /f /s /q "%~1\*.*"  >nul 2>&1
    rd /s /q  "%~1" >nul 2>&1
goto:eof

:remove_space
:remove_left_space
if "%tmp_var:~0,1%"==" " (
    set "tmp_var=%tmp_var:~1%"
    goto remove_left_space
)

:remove_right_space
if "%tmp_var:~-1%"==" " (
    set "tmp_var=%tmp_var:~0,-1%"
    goto remove_left_space
)
goto:eof
