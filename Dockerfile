FROM python:3.14-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py train.py index.html streamlit_app.py ./

# Train during the image build so the deployed container needs no dataset download.
RUN python train.py && rm -rf mlruns mlflow.db

EXPOSE 5000

CMD ["python", "app.py"]