FROM tensorflow/tensorflow:latest-gpu-jupyter

# Set working directory
WORKDIR /app

# Install system dependencies and certificates
RUN apt-get update && apt-get install -y \
    python3-pip \
    graphviz \
    ca-certificates \
    && update-ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies with trusted host

RUN pip3 install --no-cache-dir --trusted-host pypi.org --trusted-host files.pythonhosted.org -r requirements.txt

# Copy your notebook files
COPY *.ipynb .

# Expose port for Jupyter
EXPOSE 8888

# Start Jupyter notebook
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]