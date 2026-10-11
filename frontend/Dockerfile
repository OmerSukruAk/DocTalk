# DocTalk web UI. Its own image and container: it holds no backend code and no secrets,
# and reaches the API over HTTP only (API_URL).
FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PORT=8050 API_URL=http://api:8000
WORKDIR /srv
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN useradd -u 10001 -m doctalk
USER doctalk
EXPOSE 8050
HEALTHCHECK --interval=30s --timeout=3s CMD python -c "import os,urllib.request;urllib.request.urlopen(f'http://localhost:{os.environ[\"PORT\"]}/healthz')"
# Long timeout: an answer (retrieve + generate + verify) can take tens of seconds.
CMD gunicorn app:server --bind 0.0.0.0:${PORT} --workers 2 --threads 4 --timeout 200
