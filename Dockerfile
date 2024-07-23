## Dockerfile
FROM python:3.11-slim

## Set the working directory in the container
WORKDIR /app

## Copy the Pipfile and Pipfile.lock to the Docker container
COPY [ "Pipfile", "Pipfile.lock", "./" ]
## Copy your script file to the Docker container
COPY "pycode/predict_batch_s3.py" "/app/predict_batch_s3.py"
## Copy the model directory into the container at /app/model
# COPY ["model/", "model/"]

## Install pipenv
RUN pip install pip pipenv -U
## Install the dependencies using pipenv
RUN pipenv install --system --deploy

# Set the command to run your script with default arguments
ENTRYPOINT ["python", "predict_batch_s3.py"]
CMD ["2023", "8"]
