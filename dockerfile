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
# Copy dashboard and results files
COPY dashboard.py .
COPY all_model_results.json .
COPY model_structures model_structures

# Expose port for Jupyter and Streamlit
EXPOSE 8888
EXPOSE 8501

# Start Jupyter notebook by default (can override with docker run ...)
# CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]

# Start Streamlit app
# CMD ["streamlit", "run", "dashboard.py", "--server.address=0.0.0.0"]
