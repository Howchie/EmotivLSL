@echo off
setlocal
cd /d "%~dp0"

python -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)"
if errorlevel 1 (
    echo Python 3.11 or newer is required. Install it from https://www.python.org/downloads/windows/
    pause
    exit /b 1
)

python -m pip --version >nul 2>&1
if errorlevel 1 (
    echo Python pip is not available. Reinstall Python and enable the pip option.
    pause
    exit /b 1
)

python -m pipenv --version >nul 2>&1
if errorlevel 1 (
    echo Pipenv is not installed. Installing it for the current user...
    python -m pip install --user pipenv
    if errorlevel 1 (
        echo Could not install Pipenv. Check your internet connection and try again.
        pause
        exit /b 1
    )
)

echo Checking and installing the locked Python dependencies...
python -m pipenv sync --dev
if errorlevel 1 (
    echo Dependency installation failed. Check the error above and try again.
    pause
    exit /b 1
)

python -m pipenv run python main_all.py --client-id mJpALzbxF3J6eAbiz9GPNnmJOkTz7f7COgBnFgLh --client-secret tVhpfg716GTkH0dhkBr0rznta0Su8QGwYKohxDO9F7Ya2FmS3hkJFOHXesuMjXCc59zIeXp2ARl1D9eF4aoSzt25jy1AUZftsri07VkEHCp2HX6WkalQ2cdBWv578TVB --streams dev eq pow
pause
