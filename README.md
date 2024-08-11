
# Advanced NLP-Driven Content Retrieval System Language Model 

## Overview

This project demonstrates a complete pipeline for web scraping, document storage, and query processing using a Retrieval-Augmented Generation (RAG) model. The application leverages FastAPI for building RESTful APIs, Docker for containerization, and Milvus as a vector database for efficient similarity search. It also integrates Google Gemini API for advanced text generation.

## Features

- **Web Scraping**: Fetch and parse content from specified web pages.
- **Document Storage**: Store parsed content in a Milvus vector database.
- **RAG Model**: Implement a Retrieval-Augmented Generation model to process and answer user queries.
- **Google Gemini API**: Utilize Google’s generative AI capabilities for enhanced query responses.
- **RESTful API**: Provide endpoints for loading content and querying the database.
- **Containerization**: Use Docker for consistent environment setup and deployment.
- **Postman Integration**: Test API endpoints using Postman.

## Technologies Used

- **Python**: Programming language for development.
- **FastAPI**: Framework for building the RESTful API.
- **Uvicorn**: ASGI server for serving the FastAPI application.
- **Milvus**: Vector database for efficient similarity search.
- **LangChain**: Framework for document processing and RAG model integration.
- **Google Gemini API**: For generative AI capabilities.
- **Docker**: Containerization platform to package and deploy the application.
- **Postman**: Tool for testing API endpoints.

## Getting Started

### Prerequisites

- **Docker**: Ensure Docker is installed on your machine. You can download it from [Docker's official website](https://www.docker.com/get-started).
- **Python**: Python 3.9 or higher.

### Setup

1. **Clone the Repository**

   ```bash
   git clone https://github.com/iamsandeeprSand/WebScraping_RAG_Model-and-RESTful-API.git
   cd WebScraping_RAG_Model-and-RESTful-API
   ```

2. **Create a `.env` File**

   Create a `.env` file in the project root with the following content:

   ```env
   GOOGLE_API_KEY=your-google-api-key
   ```

3. **Build the Docker Image**

   ```bash
   docker build -t my-fastapi-app .
   ```

4. **Run the Docker Container**

   ```bash
   docker run -p 8000:8000 --env-file .env my-fastapi-app
   ```

   The application will be accessible at `http://localhost:8000`.

### API Endpoints

#### `/load`

- **Method**: POST
- **Description**: Load and parse content from a specified URL into the Milvus database.
- **Request Body**:

  ```json
  {
    "url": "https://en.wikipedia.org/wiki/Cricket"
  }
  ```

- **Response**:

  ```json
  {
    "message": "Content loaded successfully"
  }
  ```

#### `/query`

- **Method**: POST
- **Description**: Process a user query using the loaded content and return an answer.
- **Request Body**:

  ```json
  {
    "question": "Explain Sachin?"
  }
  ```

- **Response**:

  ```json
  {
    "answer": "Sachin Tendulkar is an Indian cricketer widely regarded as one of the greatest batsmen in the history of cricket. He holds numerous records, including the most runs in international cricket. Thanks for asking!"
  }
  ```

## Docker Setup

### Dockerfile

The `Dockerfile` defines the environment and setup for the application:

```Dockerfile
# Use the official Python image from the Docker Hub
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Expose the port that FastAPI will run on
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Docker Compose (Optional)

For managing multiple services, you can use `docker-compose`. Create a `docker-compose.yml` file:

```yaml
version: '3.8'

services:
  fastapi:
    build: .
    ports:
      - "8000:8000"
    env_file:
      - .env

  milvus:
    image: milvusdb/milvus:latest
    ports:
      - "19530:19530"
```

Run with:

```bash
docker-compose up
```

## Testing with Postman

To test the API:

1. **Load Content**: 
   - Open Postman and create a new POST request to `http://localhost:8000/load`.
   - Set the request body to:

     ```json
     {
       "url": "https://en.wikipedia.org/wiki/Cricket"
     }
     ```

   - Send the request and check the response.

2. **Query**:
   - Create a new POST request to `http://localhost:8000/query`.
   - Set the request body to:

     ```json
     {
       "question": "Explain Sachin?"
     }
     ```

   - Send the request and check the response.

## Contributing

Feel free to fork the repository and submit pull requests. Please ensure that your code adheres to the project’s coding standards and includes appropriate tests.

## Acknowledgements

- **FastAPI**: For providing an easy way to build APIs.
- **Milvus**: For scalable and efficient vector search.
- **LangChain**: For advanced document processing and RAG capabilities.
- **Google Gemini API**: For generative AI capabilities.
