# Use an official Python runtime as a base image
FROM python:3.9

# Set the working directory in the container
WORKDIR /app

# Copy all files from local folder to container
COPY . /app

# Install dependencies and system libraries
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libsm6 \
    libxext6 \
    libxrender-dev

# Install Python dependencies
RUN pip install --no-cache-dir flask flask-cors tensorflow numpy pillow joblib opencv-python gdown

# Expose the port Flask runs on
EXPOSE 8080

# Command to run the app
CMD ["python", "/app/flask_app/flask_api.py"]
