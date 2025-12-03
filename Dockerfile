FROM python:3.12-slim

# Set working directory
WORKDIR /app

# update packages, install sqlite, delete temporary files
RUN apt-get update && apt-get install -y \
    sqlite3 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p /app/data /app/logs /app/results

# Set Python to run in unbuffered mode for better logging - flush to stdout/stderr, logs appear # immediately
ENV PYTHONUNBUFFERED=1

# Default command- defines what happens when container starts
CMD ["python", "run_experiment.py", "--help"]
