@echo off
REM This script deletes all .gz files in all subfolders starting from the current directory.

echo Deleting all .gz files in all subfolders...
del /s /q *.gz

echo Deleting all .tar files in all subfolders...
del /s /q *.tar

echo Deletion complete.
pause
