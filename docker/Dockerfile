# Foundation Model Challenge for Ultrasound Image Analysis (FMC_UIA) - Docker Configuration

FROM pytorch/pytorch:2.2.2-cuda12.1-cudnn8-runtime

# Set working directory
WORKDIR /app

# Copy and install dependencies first
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all project files (.dockerignore will automatically filter unnecessary files)
COPY . .

# Create writable home directory (required by some libraries)
RUN mkdir /myhome && chmod 777 /myhome
ENV HOME=/myhome

# Ensure /app directory is writable
RUN chmod -R 777 /app

# Set Python as default entry point
ENTRYPOINT ["python"]

# Default run model.py
CMD ["model.py"]

