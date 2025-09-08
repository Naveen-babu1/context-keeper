#!/bin/bash

echo "🚀 Context Keeper Setup Script"
echo "=============================="

# Check Python version
python_version=$(python3 --version 2>&1 | grep -Po '(?<=Python )\d+\.\d+')
echo "✓ Python version: $python_version"

# Check Docker
if command -v docker &> /dev/null; then
    echo "✓ Docker is installed"
else
    echo "✗ Docker is not installed. Please install Docker first."
    exit 1
fi

# Check Docker Compose
if command -v docker-compose &> /dev/null; then
    echo "✓ Docker Compose is installed"
else
    echo "✗ Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

echo ""
echo "Starting infrastructure services..."
cd docker
docker-compose up -d
cd ..

echo ""
echo "Setting up Python environment..."
cd backend
python -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

echo ""
echo "✅ Setup complete!"
echo ""
echo "To start the API server:"
echo "  cd backend"
echo "  source venv/bin/activate"
echo "  python -m app.main"
echo ""
echo "API will be available at:"
echo "  http://localhost:8000"
echo "  http://localhost:8000/api/docs (Swagger UI)"
