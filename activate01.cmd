@echo off
:: 1. Check if the venv folder exists, if not, create it
if not exist .myVenv (
    echo Creating virtual environment...
    python -m venv .myVenv
)

:: 2. Activate the environment
call .myVenv\Scripts\activate

:: 3. Optional: Ensure torch is installed
:: pip install torch

:: 4. Keep the window open and ready for commands
echo Environment ".myVenv" is active!
cmd /k