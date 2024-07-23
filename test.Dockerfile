## test.Dockerfile
## Use the official Python image from the Docker Hub
FROM python:3.11-slim

## Set the working directory
WORKDIR "/app"

## Copy the dependencies file to the working directory
COPY [ "Pipfile", "Pipfile.lock", "./" ]
## Install dependencies
RUN pip install -U pip pipenv
RUN pipenv install --system --deploy

## Copy the rest of the application code to the working directory
COPY [ "model/", "model/" ]
COPY [ "pycode/", "pycode/" ]
COPY [ "tests/", "tests/" ]
COPY [ "integration_test/", "integration_test/" ]

# Set the command to run your script with default arguments
ENTRYPOINT [ "python" ]
CMD [ "integration_test/integration_test.py" ]
# CMD [ "pycode/predict_batch_S3.py", "2023", "8" ]
