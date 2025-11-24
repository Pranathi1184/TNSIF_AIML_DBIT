# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file first to leverage Docker cache
COPY requirements.txt ./requirements.txt

# Install dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the Streamlit app script and the trained model pipeline
COPY app_streamlit.py ./app.py
# --- MODIFICATION: Copy the correct Random Forest model file ---
COPY random_forest_model.joblib ./random_forest_model.joblib
# -------------------------------------------------------------

# Expose the default Streamlit port
EXPOSE 8501

# Healthcheck for Streamlit apps
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Command to run the Streamlit app when the container starts
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=9000", "--server.address=0.0.0.0"]