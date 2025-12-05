FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY realtime_inference_api.py ./realtime_inference_api.py
COPY model ./model
EXPOSE 80
CMD ["uvicorn", "realtime_inference_api:app", "--host", "0.0.0.0", "--port", "80"]
