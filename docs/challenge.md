# Project Documentation

## Overview

This document provides all necessary information about the machine learning project, including the rationale behind model choice, use of Docker for testing, and the setup for continuous integration and deployment with GitHub Actions and Google Cloud Platform.

## Model Selection

### XGBoost Classifier

For this project, we chose XGBoost Classifier over Logistic Regression due to its superior performance in terms of recall and F1 score. Despite the accuracy being around 55%, the choice was made to prioritize the model's ability to correctly identify the positive class, which is critical for the problem at hand.

Key Points:

- Recall and F1 score were prioritized metrics.
- XGBoost provided better performance on these metrics despite lower accuracy.

## Containerization with Docker

### Dockerfile and Docker Compose

The application was containerized using Docker, leveraging both Dockerfile and Docker Compose to ensure consistent testing environments.

Key Points:

- Dockerfile contains all the necessary steps to build the application's image.
- Docker Compose is used to define and run multi-container Docker applications, making local testing convenient and replicable.

## Continuous Integration and Deployment

### GitHub Workflows

CI/CD pipelines were set up using GitHub Workflows. The configuration files for continuous integration and continuous delivery/deployment (`ci.yml` and `cd.yml`) are included in the repository to automate the testing, building, and deployment processes.

Key Points:

- The `ci.yml` file manages the continuous integration process, including automated tests to validate changes.
- The `cd.yml` file manages the continuous delivery/deployment, automating the deployment to Google Cloud Platform.

## Google Cloud Platform Configuration

### Docker, Artifact Registry, and Compute Engine

The deployment setup on GCP involves Docker, Artifact Registry, and Compute Engine.

Key Points:

- Docker containers are used to deploy the application, ensuring consistency across different environments.
- Google Artifact Registry stores the Docker images, making them available for deployment.
- Google Compute Engine hosts the application, allowing it to be scaled and managed effectively.

## Conclusion

It is important to acknowledge the challenges encountered in the model's predictive capabilities. Despite the model's theoretical ability to predict, we faced difficulties when attempting to make predictions with new data. This was primarily due to complications in handling features that were not included in the `top_ten_features` list. As a result, our API tests could not be successfully completed with the new data, which pointed to a need for further refinement in the feature handling process during the prediction phase.

Going forward, the focus will be on enhancing the model's preprocessing pipeline to dynamically adjust to new features, thereby increasing its flexibility and adaptability in a production environment. By addressing these concerns, the model will not only predict effectively with the current dataset but also to extend our it's robustness to new and varied data sources.
