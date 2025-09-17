# Use an official TensorFlow base image with GPU support
FROM tensorflow/tensorflow:latest-gpu

# Set the working directory inside the container
WORKDIR /app

# Install system-level dependencies.
RUN apt-get update && apt-get install -y git

# Copy dependency file first
COPY requirements.txt .

# Update pip, then install from requirements.txt
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# --- NEW INVESTIGATIVE STEPS ---
# 1. List every single package that was installed and print it to the build log.
#    This will be our ground truth.
RUN echo "--- INSTALLED PACKAGES ---" && \
    pip list

# 2. Attempt to import the problematic modules during the build.
#    If this step fails, the entire build will fail, and we will see the
#    exact traceback in the Cloud Build logs.
RUN echo "--- TESTING IMPORTS ---" && \
    python -c "import google.cloud.aiplatform; print('Successfully imported google.cloud.aiplatform')" && \
    python -c "import hypertune; print('Successfully imported hypertune')"

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