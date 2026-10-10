FROM python:3.12-slim
RUN pip install --no-cache-dir fastapi "uvicorn[standard]" "pyjwt[crypto]"
WORKDIR /srv
COPY main.py .
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "9000"]
