# Use an official TensorFlow base image with GPU support
FROM tensorflow/tensorflow:latest-gpu

# Set the working directory inside the container
WORKDIR /app

# Copy dependency files first to leverage Docker layer caching
COPY requirements.txt .

# Install DVC and the Python dependencies
# We add google-cloud-aiplatform for the 'hypertune' library
RUN pip install --no-cache-dir dvc[gcs] google-cloud-aiplatform && \
    pip install --no-cache-dir -r requirements.txt

# Copy all the necessary project files and directories
COPY scripts/ scripts/
COPY run.sh .
COPY .dvc/ .dvc/
COPY data.dvc .

# Make the runner script executable
RUN chmod +x run.sh

# Set the entrypoint for the container. This command will be run
# when Vertex AI starts the container.
ENTRYPOINT ["./run.sh"]