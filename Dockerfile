# Use an official TensorFlow base image with GPU support
FROM tensorflow/tensorflow:latest-gpu

# Set the working directory inside the container
WORKDIR /app

# Install system-level dependencies.
RUN apt-get update && apt-get install -y git

# Copy dependency file first
COPY requirements.txt .

# Update pip, then install from our clean requirements.txt
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy all the necessary project files and directories
COPY scripts/ scripts/
COPY run.sh .
COPY .dvc/ .dvc/
COPY data.dvc .

# Initialize an empty git repository inside the container.
RUN git init

# Make the runner script executable
RUN chmod +x run.sh

# Set the entrypoint for the container
ENTRYPOINT ["./run.sh"]