# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file first to leverage Docker cache
COPY requirements.txt ./requirements.txt

# Install dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the Streamlit app script and the trained model
COPY app_streamlit.py ./app.py
# --- Ensure this model file name matches exactly what you saved ---
COPY manufacturing_output_model.pkl ./manufacturing_output_model.pkl
# ----------------------------------------------------------------

# Expose the default Streamlit port
EXPOSE 8501

# Healthcheck for Streamlit apps
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Command to run the Streamlit app when the container starts
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]