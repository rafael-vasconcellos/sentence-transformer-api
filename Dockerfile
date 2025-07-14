FROM python:3.10-slim
WORKDIR /app

RUN pip install --no-cache-dir -r requirements.txt
RUN python download_model.py

EXPOSE 8000

CMD ["python", "/app/server.py"]