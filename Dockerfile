# Use an official Python runtime as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy only the requirements file first to leverage Docker cache
COPY requirements.txt /app/requirements.txt

# Install any needed packages specified in requirements.txt
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Copy the rest of the application code
COPY . /app

# Set environment variables
ENV OPENAI_API_KEY=sk-G1_wkZ37sEmY4eqnGdcNig
ENV OPENAI_BASE_URL=https://dekallm.cloudeka.ai/v1

# Expose the port the app runs on
EXPOSE 6060

# Copy the start script into the container
COPY start.sh /app/start.sh

# Make the start script executable
RUN chmod +x /app/start.sh

# Run the start script
CMD ["/app/start.sh"]