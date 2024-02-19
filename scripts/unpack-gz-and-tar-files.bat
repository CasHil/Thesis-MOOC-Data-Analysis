@echo off
setlocal enableextensions disabledelayedexpansion

@REM Unzip all .gz and .tar files in all subfolders starting from the current directory.
@REM This script requires 7-Zip to be installed. You can download it from https://www.7-zip.org/download.html.

cd ..

set "pathTo7Zip=C:\Program Files\7-Zip\"
set "rootDir=."

for /r "%rootDir%" %%i in (*.gz) do (
    "%pathTo7Zip%7z.exe" x "%%i" -o"%%~dpi"
)

for /r "%rootDir%" %%i in (*.tar) do (
    "%pathTo7Zip%7z.exe" x "%%i" -o"%%~dpi"
)

echo Process completed.
pause