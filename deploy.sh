#!/bin/bash
set -e

# Define variables
REMOTE_DIR="/home/$EC2_USERNAME/concrete-tech-question-agent"
ENV_FILE="$REMOTE_DIR/.env"

echo "Starting deployment to EC2 instance..."

# Create deployment package
echo "Creating deployment package..."
zip -r deployment.zip app.py requirements.txt README.md .gitignore

# Copy files to EC2 instance
echo "Copying files to EC2 instance..."
ssh $EC2_USERNAME@$EC2_HOST "mkdir -p $REMOTE_DIR"
scp deployment.zip $EC2_USERNAME@$EC2_HOST:$REMOTE_DIR/

# SSH into EC2 and deploy
echo "Deploying application on EC2..."
ssh $EC2_USERNAME@$EC2_HOST << 'EOF'
  cd $REMOTE_DIR
  unzip -o deployment.zip
  
  # Create virtual environment if it doesn't exist
  if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
  fi
  
  # Activate virtual environment and install dependencies
  source venv/bin/activate
  pip install --upgrade pip
  pip install -r requirements.txt
  
  # Create or update .env file with secrets
  echo "GOOGLE_API_KEY=${GOOGLE_API_KEY}" > .env
  
  # Create or update transcript file with empty content if not exists
  if [ ! -f "cleaned_transcript.txt" ]; then
    echo "Creating empty transcript file..."
    touch cleaned_transcript.txt
  fi
  
  # Create systemd service file for the application if it doesn't exist
  if [ ! -f "/etc/systemd/system/concrete-tech-agent.service" ]; then
    echo "Creating systemd service..."
    sudo bash -c 'cat > /etc/systemd/system/concrete-tech-agent.service << EOL
[Unit]
Description=Concrete Tech Question Agent Service
After=network.target

[Service]
User=$USER
WorkingDirectory=$REMOTE_DIR
ExecStart=$REMOTE_DIR/venv/bin/gunicorn --bind 0.0.0.0:8000 app:app
Restart=always
Environment="PATH=$REMOTE_DIR/venv/bin"
Environment="GOOGLE_API_KEY=${GOOGLE_API_KEY}"

[Install]
WantedBy=multi-user.target
EOL'

    sudo systemctl daemon-reload
    sudo systemctl enable concrete-tech-agent.service
  fi
  
  # Restart the service
  sudo systemctl restart concrete-tech-agent.service
  
  # Clean up
  rm deployment.zip
  
  echo "Deployment completed successfully!"
EOF

echo "Deployment process finished!"