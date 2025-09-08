#!/bin/bash
cd /d/projects/context-keeper/backend
source venv/Scripts/activate
echo "Starting Context Keeper API Server..."
uvicorn app.main:app --host 0.0.0.0 --port 8000
