# amazon-sagemaker-examples

## Amazon SageMaker Overview

[Amazon SageMaker](https://aws.amazon.com/sagemaker/) is a fully managed service for data science and machine learning (ML) workflows.
You can use Amazon SageMaker to simplify the process of building, training, and deploying ML models.

## :hammer_and_wrench: Amazon SageMaker Setup

The quickest setup to run example notebooks includes:
- An [AWS account](http://docs.aws.amazon.com/sagemaker/latest/dg/gs-account.html)
- Proper [IAM User and Role](http://docs.aws.amazon.com/sagemaker/latest/dg/authentication-and-access-control.html) setup
- An [Amazon SageMaker Notebook Instance](http://docs.aws.amazon.com/sagemaker/latest/dg/gs-setup-working-env.html)
- An [S3 bucket](http://docs.aws.amazon.com/sagemaker/latest/dg/gs-config-permissions.html)

## Amazon SageMaker Examples

These examples provide an introduction to machine learning concepts as well as SageMaker.

- [Heart Failure Prediction](heartdisease) predicts congestive heart failure based on patient data such as age, gender, smoking, blood pressure, and other biochemical markers.  The patient [data set](https://archive.ics.uci.edu/ml/datasets/Heart+failure+clinical+records) consists of 299 anonymized records sourced from the University California at Irvine Machine Learning Repository.  This example uses a number of SageMaker components for building, tuning, tracking, and deploying ML models and it demonstrates several ML algorithms including XGBoost, Linear Learner, and K-Nearest Neighbor.
