#!/bin/bash
# Production Deployment Script for RegretZero PPO
# This script helps deploy the RegretZero PPO application to production

set -e

echo "🚀 Starting RegretZero PPO Deployment..."
echo "=================================="

# Check if model exists
if [ ! -f "model/regret_ppo.pt" ]; then
    echo "⚠️  Warning: No trained model found at model/regret_ppo.pt"
    echo "   The application will run with untrained PPO agent"
    echo "   Consider training a model first: python model/train_ppo.py"
    echo ""
fi

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    echo "   Visit: https://docs.docker.com/get-docker/"
    exit 1
fi

# Check if Docker is running
if ! docker info &> /dev/null; then
    echo "❌ Docker daemon is not running. Please start Docker first."
    exit 1
fi

# Build Docker image
echo "📦 Building Docker image..."
docker build -t regretzero-ppo .

# Run Docker container
echo "🏃 Running Docker container..."
docker run -d \
    --name regretzero-ppo \
    -p 8000:8000 \
    -e PORT=8000 \
    --restart unless-stopped \
    regretzero-ppo

echo ""
echo "✅ Deployment complete!"
echo "🌐 RegretZero PPO is running at: http://localhost:8000"
echo "📊 Health check: http://localhost:8000/health"
echo "📖 API docs: http://localhost:8000/docs"
echo ""
echo "🛑 To stop: docker stop regretzero-ppo"
echo "🔄 To restart: docker restart regretzero-ppo"
echo "📋 To view logs: docker logs regretzero-ppo"
echo ""
echo "🎯 PPO Model Status:"
if [ -f "model/regret_ppo.pt" ]; then
    echo "   ✅ Trained model loaded"
else
    echo "   ⚠️  Running with untrained model"
fi
