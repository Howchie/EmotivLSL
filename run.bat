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

set "LAUNCH_SCRIPT=main_all_epochX.py"
set "LAUNCH_ARGS=--streams dev eq pow"
if /I "%~1"=="flex" (
    set "LAUNCH_SCRIPT=main_all_flex.py"
    set "LAUNCH_ARGS=--streams dev eq --mapping epoch_flex_electrodes.json --remove-dc"
)

python -m pipenv run python %LAUNCH_SCRIPT% --client-id 3hQag0PH5ruAjQ6sWmkC3OPZs4EHL8Rmz1W92g3Z --client-secret 0UmT695g4xIu96M2zk9NAu4vB0KTPPX5MhnGBV1EFNNYeyB1OJf0h9TMLa5BB9gZibuegusB6Tc0nZWhJTkay7Ed629nlHRiq974jCjA8rdRgKh98uFSB4361Z4ehzpg %LAUNCH_ARGS%
pause
