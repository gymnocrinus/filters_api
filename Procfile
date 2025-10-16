web: gunicorn -k uvicorn.workers.UvicornWorker app.main:app -w 1 --timeout 120 --max-requests 500 --max-requests-jitter 50
