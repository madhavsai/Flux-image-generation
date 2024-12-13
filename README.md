# Image Generation Application

This repository contains an image generation application that utilizes a modified version of the Flux model. The backend API services are implemented using Flask, and the deployment is managed with Docker. A Dockerfile is also provided for easy setup and deployment.

## Features
- **Modified Flux Model**: Utilizes a custom version of the Flux model for image generation.
- **Backend API**: Built using Flask for handling image generation requests.
- **Dockerized Deployment**: Includes a Dockerfile for containerized deployment.

## Getting Started

### Prerequisites
- Docker installed on your machine.
- Python 3.7 or higher.

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/YourUsername/Image-generation-with-Flux.git
   cd Image-generation-with-Flux
Build the Docker image:

bash
docker build -t image-generation-app .
Run the Docker container:

bash
docker run -p 5000:5000 image-generation-app
Usage
Start the Flask API: The application runs on http://localhost:5000. You can send POST requests to the /generate-image-single or /generate-image endpoints with the necessary parameters to generate images.

Example Requests:

Single Image Generation:

bash
curl -X POST http://localhost:5000/generate-image-single -d '{"prompt": "sunset over mountains", "width": 512, "height": 512}' -H "Content-Type: application/json"
Multiple Images Generation:

bash
curl -X POST http://localhost:5000/generate-image -d '{"prompt": "sunset over mountains", "width": 512, "height": 512}' -H "Content-Type: application/json"
API Endpoints
GET /:

Description: Serves the index page.

Response: Renders the index.html template.

POST /generate-image-single:

Description: Generates a single image based on the provided prompt.

Request Parameters:

prompt (string, required): A textual description of the desired image.

width (integer, optional): The width of the generated image (default is 512).

height (integer, optional): The height of the generated image (default is 512).

Response: JSON containing a message and the URL of the generated image.

POST /generate-image:

Description: Generates four images based on the provided prompt.

Request Parameters:

prompt (string, required): A textual description of the desired image.

width (integer, optional): The width of the generated images (default is 512).

height (integer, optional): The height of the generated images (default is 512).

Response: JSON containing a message and a list of URLs for the generated images.

Dockerfile
dockerfile
# Use an official Python runtime as a parent image
FROM python:3.8-slim-buster

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Make port 5000 available to the world outside this container
EXPOSE 5000

# Define environment variable
ENV FLASK_ENV=development

# Run app.py when the container launches
CMD ["python", "app.py"]
Contributing
Feel free to fork this repository and submit pull requests. For major changes, please open an issue first to discuss what you would like to change.
