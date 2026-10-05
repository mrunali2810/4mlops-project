FROM python:3.11-slim
WORKDIR /app
RUN pip install --no-cache-dir pandas scikit-learn joblib
COPY src/ src/
COPY models/ models/
CMD ["python", "src/predict.py", "5.1", "3.5", "1.4", "0.2"]
