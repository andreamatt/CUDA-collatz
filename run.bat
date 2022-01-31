@echo off

nvcc .\%1.cu -o .\build\%1.exe
.\build\%1.exe