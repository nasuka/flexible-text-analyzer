#!/bin/bash

# Azure App Service startup script for mirai-analyzer
echo "Starting mirai-analyzer on Azure App Service..."

# Install system dependencies for MeCab
apt-get update
apt-get install -y mecab mecab-ipadic-utf8

# Set environment variables
export PYTHONPATH="${PYTHONPATH}:/home/site/wwwroot"

# Start the Streamlit application
cd /home/site/wwwroot
python -m streamlit run src/app.py --server.port=8000 --server.address=0.0.0.0 --server.enableCORS=false --server.enableXsrfProtection=false