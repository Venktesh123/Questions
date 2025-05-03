#!/bin/bash

# This script can be added to EC2 user data to start the service on instance boot
# It ensures the service starts even after instance stop/start cycles

# Wait for system to fully initialize
sleep 30

# Start and enable the service
systemctl daemon-reload
systemctl enable concrete-tech-agent.service
systemctl start concrete-tech-agent.service

# Log the status
echo "Service startup attempted at $(date)" >> /var/log/concrete-tech-agent-startup.log
systemctl status concrete-tech-agent.service >> /var/log/concrete-tech-agent-startup.log