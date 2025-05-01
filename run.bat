@echo off
cd /d "%~dp0"
echo Starting Streamlit app deployment...

IF NOT EXIST .venv\Scripts\python.exe (
  echo Creating virtual environment...
  python -m venv .venv
  echo Upgrading pip...
  .venv\Scripts\python.exe -m pip install --upgrade pip
  echo Installing dependencies...
  .venv\Scripts\pip install -r requirements.txt
) ELSE (
  echo Virtual environment exists. Using existing environment.
)

echo Activating virtual environment...
call .venv\Scripts\activate.bat

echo Starting Streamlit app on port %HTTP_PLATFORM_PORT%...
streamlit run app.py --server.port %HTTP_PLATFORM_PORT% --server.enableCORS=false --server.enableXsrfProtection=false