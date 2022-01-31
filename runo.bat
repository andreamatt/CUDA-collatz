@echo off

nvcc .\%1.cu -O2 -o .\build\%1.exe
.\build\%1.exe