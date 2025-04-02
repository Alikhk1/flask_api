# Use an official Python runtime as a base image
FROM python:3.9

# Set the working directory in the container
WORKDIR /app

# Install necessary dependencies
RUN pip install --no-cache-dir flask flask-cors tensorflow-cpu numpy pillow joblib opencv-python gdown

# Install libGL for OpenCV
RUN apt-get update && apt-get install -y libgl1-mesa-glx

# Copy all files from local folder to container
COPY . /app

# Download model from Google Drive
RUN gdown --id 1-0E4AQHqYJdJxqx0MSWb1uNVxtDZh78k -O /app/flask_app/model.tflite

# Expose the port Flask runs on
EXPOSE 8080

# Command to run the app
CMD ["python", "/app/flask_app/flask_api.py"]
