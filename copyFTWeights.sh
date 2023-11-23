#!/bin/bash

# Variables
REMOTE_HOST="80.124.71.136"
REMOTE_PORT="12829"
REMOTE_USER="root"
PRIVATE_KEY_FILE="gpu" 
REMOTE_DIR="/workspace/lora/output/*" 
LOCAL_DIR="."

# Copy files from remote server to local directory
scp -i "$PRIVATE_KEY_FILE" -P "$REMOTE_PORT" "$REMOTE_USER@$REMOTE_HOST:$REMOTE_DIR" "$LOCAL_DIR"