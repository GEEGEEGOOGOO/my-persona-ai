# Start with a specific, known-stable version of Python
FROM python:3.11-slim

# Set the working directory
WORKDIR /app

# Copy the requirements file
COPY requirements.txt ./

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application files
COPY . .

# This command makes the script executable
RUN chmod +x ./start.sh

# Create a startup script to handle the port correctly
# This is the most reliable way to handle the PORT variable
RUN echo '#!/bin/sh' > ./start.sh && \
    echo 'streamlit run webapp.py --server.port=$PORT --server.address=0.0.0.0' >> ./start.sh

# Set the command to run the startup script
CMD ["./start.sh"]
