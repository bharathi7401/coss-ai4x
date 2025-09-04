# Use an official lightweight Python image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Copy requirements first (for caching)
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy your app
COPY . .

# Expose port 8000 for Prometheus scraping
EXPOSE 8000

# Run app (adjust depending on framework)
CMD ["python", "ai4x_demo_app.py"]
