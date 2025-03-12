
# Use an official Python runtime as a base image
FROM python:3.9

# Set the working directory in the container
WORKDIR /app

# Copy all files from local folder to container
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir flask flask-cors tensorflow numpy pillow

# Expose the port Flask runs on
EXPOSE 8080

# Command to run the app
CMD ["python", "/flask_app/flask_api.py"]
