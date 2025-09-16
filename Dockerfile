# Use an official TensorFlow base image with GPU support
FROM tensorflow/tensorflow:latest-gpu

# Set the working directory inside the container
WORKDIR /app

# Copy dependency file first to leverage Docker layer caching
COPY requirements.txt .

# Update pip, then install from requirements.txt with no cache
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy all the necessary project files and directories
COPY scripts/ scripts/
COPY run.sh .
COPY .dvc/ .dvc/
COPY data.dvc .

# --- NEW CRITICAL FIX ---
# Initialize an empty git repository inside the container.
# This gives DVC the context it needs to run without the --no-scm flag.
RUN git init

# Make the runner script executable
RUN chmod +x run.sh

# Set the entrypoint for the container
ENTRYPOINT ["./run.sh"]