## Makefile
#### This Makefile contains various targets for project management tasks such as running the project,
### cleaning up build files, running tests, building Docker images, and more.
## Phony targets are used to avoid conflicts with files of the same name.

## Declare phony targets to indicate these are not files but commands to be executed.
.PHONY: run setup quality_checks test build integration_test publish clean all
## To run any of these targets, use the `make` command followed by the target name.
## For example:
##   make run              # Executes the run target to run the main script or application
##   make setup            # Executes the setup target to install dependencies
##   make quality_checks   # Executes the quality_checks target to run code quality checks
##   make test             # Executes the test target to run unit tests
##   make build            # Executes the build target to run quality checks, tests, and build the Docker image
##   make integration_test # Executes the integration_test target to run integration tests
##   make publish          # Executes the publish target to build, test, and publish the Docker image
##   make clean            # Executes the clean target to remove build artifacts
##   make all              # Executes the all target to clean and build the project

## run target: Executes the commands to run the project and displays the Python version.
## Each command should be prefixed with a tab character.
run:
	echo "Running the project"	
	python -V


## setup target: Installs development dependencies and sets up pre-commit hooks.
## Run this target to prepare the environment.
# pre-commit install
setup:
	pipenv install --dev
	pipenv graph
	echo "setup completed."

## quality_checks target: Runs code quality checks using isort, black, and pylint.
## Run this target to ensure code quality.
quality_checks:
	isort . --verbose || { echo "isort failed"; exit 1; }
	black . --verbose
	pylint --recursive=y . --verbose
	echo "quality_checks completed."

## test target: Runs pytest on the tests/ directory.
## Run this target to execute unit tests.
test:
	pytest tests/
	echo "pytest completed."

## Variables for image tagging and naming
## LOCAL_TAG:=$(shell date +"%Y-%m-%d-%H-%M")  ${LOCAL_TAG}
LOCAL_IMAGE_NAME:=mushroom-test-app
## build target: Runs quality checks and tests, then builds the Docker image.
## This target depends on quality_checks and test.
build: quality_checks test
	docker build -t ${LOCAL_IMAGE_NAME} -f test.Dockerfile .
	echo "build completed."docker --version)

## integration_test target: Runs integration tests after building the Docker image.
## This target depends on the build target.
integration_test: build
	LOCAL_IMAGE_NAME=${LOCAL_IMAGE_NAME} bash integration_test/run.sh
	echo "integration_test completed."

## publish target: Builds the Docker image, runs integration tests, and publishes the image.
## This target depends on build and integration_test.
publish: integration_test
	LOCAL_IMAGE_NAME=${LOCAL_IMAGE_NAME} bash scripts/publish.sh
	echo "publish completed."

## clean target: Removes build artifacts and cleans up the project directory.
## Useful for ensuring a fresh build environment.
clean:
	mkdir -p current_dir
	rm -rf current_dir
	echo "clean completed."

# all target: A convenience target that cleans the build directory and then builds the app.
# Ensures that the project is rebuilt from a clean state.
all: clean
	# Assuming my_app is built using some other rule or command
	make my_app
	echo "all completed."
