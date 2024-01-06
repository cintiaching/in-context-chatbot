# Use the base image with Python 3.10
FROM --platform=linux/amd64 python:3.10

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file to the container
COPY requirements/requirements-document-chatbot.txt .

# Install the Python dependencies
RUN pip install --no-cache-dir -r requirements-document-chatbot.txt

# Copy the app files to the container
COPY document_chatbot_app.py .
COPY chatbots /app/document_chatbot

# Expose the port on which the Streamlit app will run
EXPOSE 8501

# Set the default command to run the Streamlit app
CMD ["streamlit", "run", "document_chatbot_app.py"]