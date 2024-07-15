# MLOps-Zoomcamp-M7-Project-Attempt-1-Mushroom-Classification
MLOps-Zoomcamp-M7-Project-Attempt-1-Mushroom-Classification

# Development Checklist

## Problem Description
- [x] Describe the problem clearly and thoroughly

## Cloud Integration
- [x] Ensure cloud services are used for the project
  - [x] Develop on the cloud OR use Localstack (or similar tool) OR deploy to Kubernetes or similar container management platforms
  - [x] Use Infrastructure as Code (IaC) tools for provisioning the infrastructure

## Experiment Tracking and Model Registry
- [x] Implement experiment tracking
- [x] Register models in a model registry

## Workflow Orchestration
- [x] Implement basic workflow orchestration
- [x] Deploy a fully orchestrated workflow

## Model Deployment
- [x] Deploy the model
  - [x] Deploy the model locally
  - [x] Containerize the model deployment code
  - [x] Ensure the model can be deployed to the cloud or special tools for model deployment are used

## Model Monitoring
- [x] Implement model monitoring
  - [x] Calculate and report basic metrics
  - [x] Set up comprehensive model monitoring that sends alerts or runs conditional workflows if metrics thresholds are violated (e.g., retraining, generating debugging dashboard, switching to a different model)

## Reproducibility
- [x] Provide clear instructions on how to run the code
  - [x] Ensure all instructions are complete and clear
  - [x] Verify the code works as described
  - [x] Include versions for all dependencies
  - [x] Ensure data is available and accessible

## Best Practices
- [x] Write unit tests
- [x] Write integration tests
- [x] Use a linter and/or code formatter
- [ ] Create a Makefile for the project
- [ ] Set up pre-commit hooks
- [ ] Implement a CI/CD pipeline
