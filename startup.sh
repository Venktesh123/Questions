#!/bin/bash
export PORT=${PORT:-8000}
echo "Starting app on port $PORT"
gunicorn --bind=0.0.0.0:$PORT app:app