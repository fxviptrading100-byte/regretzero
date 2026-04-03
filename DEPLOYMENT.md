# 🚀 RegretZero PPO Deployment Guide

## Production Deployment

This guide covers deploying the RegretZero PPO-powered decision advisor to production.

## Prerequisites

- Docker and Docker Compose installed
- Trained PPO model (`model/regret_ppo.pt`)
- 8000 port available

## Quick Start

### Option 1: Using Deploy Script (Recommended)
```bash
# Make deploy script executable (Linux/Mac)
chmod +x deploy.sh

# Run deployment
./deploy.sh
```

### Option 2: Manual Docker Deployment
```bash
# Build the Docker image
docker build -t regretzero-ppo .

# Run the container
docker run -d \
    --name regretzero-ppo \
    -p 8000:8000 \
    -e PORT=8000 \
    --restart unless-stopped \
    regretzero-ppo
```

### Option 3: Using Docker Compose
Create `docker-compose.yml`:
```yaml
version: '3.8'
services:
  regretzero-ppo:
    build: .
    container_name: regretzero-ppo
    ports:
      - "8000:8000"
    environment:
      - PORT=8000
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

Then run:
```bash
docker-compose up -d
```

## Access Points

Once deployed, access the application at:

- **Main Application**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health
- **Interactive Demo**: http://localhost:8000/demo
- **PPO Inference**: http://localhost:8000/inference

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | 8000 | Application port |
| `PYTHONUNBUFFERED` | 1 | Disable Python output buffering |
| `PYTHONDONTWRITEBYTECODE` | 1 | Disable .pyc file generation |

## Model Management

### With Trained Model
If `model/regret_ppo.pt` exists, the application will:
- Load the trained PPO model
- Provide intelligent, learned recommendations
- Show confidence scores and uncertainty estimates

### Without Trained Model
If no model is found, the application will:
- Create an untrained PPO agent
- Provide basic rule-based recommendations
- Show lower confidence scores

## Production Features

### Security
- Non-root Docker user
- Read-only file system where possible
- Health checks with automated restarts

### Performance
- Python 3.11 slim base image
- Optimized Docker layer caching
- Single worker process for resource efficiency

### Monitoring
- Structured logging
- Health check endpoints
- Graceful shutdown handling

## API Endpoints

### Core PPO Endpoints
- `POST /predict` - Get PPO decision recommendation
- `POST /batch-predict` - Multiple decision analysis
- `GET /model-info` - PPO model information

### Demo Endpoints
- `GET /demo` - Interactive decision advisor demo
- `GET /inference` - PPO inference interface
- `GET /` - Main application interface

## Troubleshooting

### Common Issues

1. **Port Already in Use**
   ```bash
   # Check what's using port 8000
   netstat -tulpn | grep :8000
   
   # Kill the process
   sudo kill -9 <PID>
   ```

2. **Model Not Loading**
   ```bash
   # Check if model file exists
   ls -la model/regret_ppo.pt
   
   # Check container logs
   docker logs regretzero-ppo
   ```

3. **Health Check Failing**
   ```bash
   # Manual health check
   curl -f http://localhost:8000/health
   
   # View detailed logs
   docker logs regretzero-ppo --tail 50
   ```

### Logs and Monitoring

```bash
# View live logs
docker logs -f regretzero-ppo

# Check container status
docker ps | grep regretzero-ppo

# Resource usage
docker stats regretzero-ppo
```

## Scaling Considerations

### Horizontal Scaling
```bash
# Run multiple instances
docker run -d --name regretzero-ppo-2 -p 8001:8000 regretzero-ppo
docker run -d --name regretzero-ppo-3 -p 8002:8000 regretzero-ppo
```

### Load Balancing
Use nginx or cloud load balancer to distribute traffic across instances.

## Development vs Production

| Feature | Development | Production |
|---------|-------------|------------|
| Debug logging | ✅ | ❌ |
| Auto-reload | ✅ | ❌ |
| Single worker | ❌ | ✅ |
| Security user | ❌ | ✅ |
| Health checks | ❌ | ✅ |

## Support

For deployment issues:
1. Check container logs: `docker logs regretzero-ppo`
2. Verify model file exists in `model/` directory
3. Ensure port 8000 is available
4. Review health check endpoint status

The PPO-powered RegretZero is now ready for production deployment! 🎯
