# backend/Dockerfile
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Copy code and install dependencies
COPY . .
RUN pip install --no-cache-dir -r requirements.txt

# Expose FastAPI default port
EXPOSE 8000

# Start FastAPI app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]