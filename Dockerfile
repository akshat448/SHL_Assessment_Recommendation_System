FROM python:3.10-slim

# Set working directory for the app
WORKDIR /app

# Set Python path to include the app and agents directories
ENV PYTHONPATH="/app:/app/agents:${PYTHONPATH:-}"

# Copy requirements and install dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt streamlit pandas together

# Copy the entire project directory
COPY . /app

# Create model cache directory
RUN mkdir -p /app/.model_cache

# Use a volume for model cache
VOLUME ["/app/.model_cache"]

# Expose ports for both FastAPI and Streamlit
EXPOSE 8080 8501

# Command to run both FastAPI and Streamlit
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port 8080 & streamlit run app.py --server.port=8501 --server.address=0.0.0.0"]