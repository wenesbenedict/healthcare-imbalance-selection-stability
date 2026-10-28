@echo off
setlocal
cd /d "%~dp0"
py demo_cli.py --training-size 250 --imbalance-ratio 5 --noise 5 --repeats 20
pause
