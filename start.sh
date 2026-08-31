#!/bin/sh
set -e

PORT="${PORT:-8080}"

envsubst '${PORT}' < /etc/nginx/nginx.conf.template > /etc/nginx/conf.d/default.conf

cd /app
uvicorn app.main:app --host 127.0.0.1 --port 8000 &
UVICORN_PID=$!

nginx -g 'daemon off;' &
NGINX_PID=$!

trap 'kill $UVICORN_PID $NGINX_PID 2>/dev/null; wait' TERM INT

wait
