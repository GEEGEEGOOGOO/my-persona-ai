# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install the Python libraries
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application files into the container
COPY . .

# Tell Docker that the container listens on port 8501
EXPOSE 8080

# The command to run when the container starts
CMD ["streamlit", "run", "webapp.py", "--server.port=8080", "--server.address=0.0.0.0"]

