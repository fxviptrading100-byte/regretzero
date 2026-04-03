# Production Procfile for RegretZero PPO Deployment
# Configures the web server for production environment
web: uvicorn env.regret_openenv:app --host 0.0.0.0 --port $PORT --workers 1 --access-log - --log-level info
