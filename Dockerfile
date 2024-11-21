# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Set environment variables
ENV OPENAI_API_KEY=sk-G1_wkZ37sEmY4eqnGdcNig
ENV OPENAI_BASE_URL=https://dekallm.cloudeka.ai/v1

# Expose the port the app runs on
EXPOSE 6060

# Run the application
CMD ["python", "-m", "routellm.openai_server", "--routers", "mf", "--strong-model", "openai/qwen/qwen2-vl-72b-instruct", "--weak-model", "openai/gotocompany/gemma2-9b-cpt-sahabatai-v1-instruct"]