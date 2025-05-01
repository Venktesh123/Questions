#!/bin/bash
# This script creates all necessary files for Azure Web App deployment

# Create web.config
cat > web.config << 'EOL'
<?xml version="1.0" encoding="utf-8"?>
<configuration>
  <system.webServer>
    <handlers>
      <add name="PythonHandler" path="*" verb="*" modules="httpPlatformHandler" resourceType="Unspecified"/>
    </handlers>
    <httpPlatform processPath="%home%\site\wwwroot\run.bat"
                  arguments=""
                  stdoutLogEnabled="true"
                  stdoutLogFile="%home%\LogFiles\python.log"
                  startupTimeLimit="600"
                  startupRetryCount="3">
      <environmentVariables>
        <environmentVariable name="PORT" value="%HTTP_PLATFORM_PORT%" />
        <environmentVariable name="STREAMLIT_SERVER_PORT" value="%HTTP_PLATFORM_PORT%" />
      </environmentVariables>
    </httpPlatform>
  </system.webServer>
</configuration>
EOL

# Create run.bat
cat > run.bat << 'EOL'
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
EOL

# Create .streamlit directory
mkdir -p .streamlit

# Create .streamlit/config.toml
cat > .streamlit/config.toml << 'EOL'
[server]
enableCORS = false
enableXsrfProtection = false
headless = true
port = 8501

[browser]
gatherUsageStats = false
serverAddress = "0.0.0.0"

[runner]
magicEnabled = true
EOL

# Create an empty .env file if it doesn't exist
if [ ! -f .env ]; then
    echo "# Add your environment variables here" > .env
    echo "GOOGLE_API_KEY=" >> .env
    echo ".env file created. Please add your API keys."
fi

# Make sure the script is executable
chmod +x deploy.sh

echo "Deployment files created successfully."
echo "Remember to set GOOGLE_API_KEY in your Azure App Service Configuration."