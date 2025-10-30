# streamlit_app/Dockerfile

FROM python:3.10-slim

# Set working directory
WORKDIR /app

# [Optimization 1: Copy requirements first to leverage build cache]
COPY requirements.txt .

# Install dependencies (This layer is cached if requirements.txt doesn't change)
RUN pip install --no-cache-dir -r requirements.txt

# [Optimization 2: Copy application code]
# Only copy the rest of the application once dependencies are installed.
COPY . .

# Expose Streamlit port
EXPOSE 8501

# Run the Streamlit app (Using full CMD array format is best practice)
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]