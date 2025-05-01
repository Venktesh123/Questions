# This PowerShell script helps with deploying to Azure
# You can run it locally to test the deployment setup

# Create web.config file
$webConfigContent = @"
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
"@

Set-Content -Path "web.config" -Value $webConfigContent

# Create run.bat
$runBatContent = @"
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
"@

Set-Content -Path "run.bat" -Value $runBatContent

# Create .streamlit directory if it doesn't exist
if (-not (Test-Path ".streamlit")) {
    New-Item -ItemType Directory -Path ".streamlit"
}

# Create .streamlit/config.toml
$configTomlContent = @"
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
"@

Set-Content -Path ".streamlit/config.toml" -Value $configTomlContent

Write-Host "Deployment files created successfully."
Write-Host "Make sure to set the GOOGLE_API_KEY in your Azure App Service Configuration."