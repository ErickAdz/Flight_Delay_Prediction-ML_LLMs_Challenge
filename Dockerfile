# syntax=docker/dockerfile:1.2
FROM python:3.10.0-slim-buster

# Set an environment variable to ensure python output is sent straight to the terminal.
ENV PYTHONUNBUFFERED 1


WORKDIR /app

# Install system dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc build-essential \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip, wheel, and setuptools to make sure we have the latest versions installed
RUN pip install --upgrade pip setuptools wheel

# Copy the local directory into the container.
COPY . /app/

RUN pip install --no-cache-dir -r requirements.txt -r requirements-dev.txt -r requirements-test.txt

# Inform Docker that the container listens on the specified network ports at runtime.
# EXPOSE 8000

# The command to run the FastAPI server using uvicorn.
CMD ["uvicorn", "challenge.api:app", "--host", "0.0.0.0", "--port", "8000"]