# Heart Failure Prediction Model using Amazon SageMaker

## :bulb: 0.0 Heart Failure Machine Learning Model Motivation

Improved data and analytical tools can assist healthcare providers in identifying patients who are at-risk from heart failure, thus allowing providers to proactively recommend and implement a holistic protocol consisting of lifestyle changes, pharmaceutical drugs, and advanced interventions (e.g. angioplasty, catheter ablation, stent placement, and surgery to repair valves and vessels).  Therefore, data analytics and machine learning (ML) can play an important part in the overall healthcare solution that saves lives from CardioVascular Diseases (CVD) and improves the quality and longevity of life.  This example project uses [Amazon SageMaker](https://aws.amazon.com/sagemaker/) which is a managed ML service provided by Amazon Web Services (AWS), and it follows the [CRISP-DM](https://en.wikipedia.org/wiki/Cross-industry_standard_process_for_data_mining) methodology in which the data analytics lifecycle consists of iterative steps:

- Business Domain Understanding
- Data Understanding
- Data Preparation
- ML Modeling
- ML Evaluation
- ML Deployment

## :hospital: 1.0 Heart Failure Domain Understanding

Cardiovascular disease (CVD) is a leading cause of death globally, taking approximately 17 million lives each year. CVDs are a group of diseases of the heart and blood vessels and include coronary heart disease, cerebrovascular disease, rheumatic heart disease and other conditions. More than four out of five CVD deaths are due to heart attacks and strokes, and one third of these deaths occur prematurely in people under 70 years of age.

The most important behavioural risk factors of heart disease and stroke are unhealthy diet, physical inactivity, tobacco use and harmful use of alcohol. The effects of behavioural risk factors may show up in individuals as raised blood pressure, raised blood glucose, raised blood lipids, and overweight. These “intermediate risks factors” can be measured in primary care facilities and indicate an increased risk of heart attack, stroke, heart failure and other complications.

Cessation of tobacco use, reduction of salt in the diet, eating more fruit and vegetables, regular physical activity and avoiding harmful use of alcohol have been shown to reduce the risk of cardiovascular disease. Public health policies that create conducive environments for making healthy choices affordable and available are essential for motivating people to adopt and sustain healthy behaviours.

Identifying those at highest risk of CVDs and ensuring they receive appropriate preventative and interventional treatments can improve quality and longevity of life as well as reduce the long-term cost of caring for these patients.

References:
- [Heart Disease WHO Source](https://www.who.int/health-topics/cardiovascular-diseases)

## :floppy_disk: 2.0 Heart Failure Data Understanding

The data set used to train the ML models in this project consists of 299 records of patients who experienced heart failure while at the Faisalabad Institute of Cardiology and at the Allied Hospital in Faisalabad during the period of April-December 2015; each record consists of thirteen (13) attributes and the target feature is a boolean attribute that denotes whether the patient survived the heart failure or not. Therefore, this ML scenario can be framed as a supervised binary classification problem such that the target feature denotes 1 for death from heart failure and 0 for survival/no failure; the goal is to predict whether new incoming patients with a set of similar attributes will survive from heart failure or not.  Also, note that the data set was sourced from the highly regarded and curated [University of California Irvine Machine Learning Repository](https://archive.ics.uci.edu/ml/index.php). 

- age: age of the patient (years)
- anaemia: decrease of red blood cells or hemoglobin (boolean)
- high blood pressure: if the patient has hypertension (boolean)
- creatinine phosphokinase (CPK): level of the CPK enzyme in the blood (mcg/L)
- diabetes: if the patient has diabetes (boolean)
- ejection fraction: percentage of blood leaving the heart at each contraction (percentage)
- platelets: platelets in the blood (kiloplatelets/mL)
- sex: woman or man (binary)
- serum creatinine: level of serum creatinine in the blood (mg/dL)
- serum sodium: level of serum sodium in the blood (mEq/L)
- smoking: if the patient smokes or not (boolean)
- time: follow-up period (days)
- [target] death event: if the patient deceased during the follow-up period (boolean)

References:
- [ML Data Source](https://archive.ics.uci.edu/ml/datasets/Heart+failure+clinical+records)
- [ML Paper by Davide Chicco and Guiseppe Jurman](https://doi.org/10.1186/s12911-020-1023-5)

## :bar_chart: 2.1 Heart Failure Data Preparation

For data exploration and preparation, I took a comprehensive approach that used Amazon Sagemaker [Data Wrangler](https://aws.amazon.com/sagemaker/data-wrangler/) as well as Jupyter notebooks applying Python libraries such as pandas, matplotlib, and seaborn to interactively explore the data set.

Amazon SageMaker Data Wrangler is a visual component for aggregating, exploring, and preparing data that simplifies the data pipeline and feature engineering process so that you can reduce the time it takes to engineer robust features for ML models from weeks and days into hours and minutes.  Data Wrangler has data connectors for S3, Athena, RedShift, Snowflake and other data sources, and it has more than 300 built-in data transforms so that you can quickly normalize, enrich, and combine features without having to write any code.  Data Wrangler enables you to also author custom transformations using Pandas, PySpark, and SQL when necessary.  Data Wrangler also has a robust set of visualization templates including correlation heatmaps, histograms, scatter plots, box and whisker plots, line charts, and bar charts that are all available to help you understand your data, identify relationships and spot outliers.  Data Wrangler has a Quick Model feature that enables you to train a simple ML model using the Random Forest algorithm and quickly diagnose issues earlier in your data pipeline before more complex models and resources are deployed into production. Data Wrangler also integrates with [SageMaker Clarify](https://aws.amazon.com/sagemaker/clarify/), making it easier to identify bias and imbalances during data preparation.  Finally, Data Wrangler workflows can be exported to a Jupyter Notebook or Python script for MLOps automation; you can also publish features to [SageMaker Feature Store](https://aws.amazon.com/sagemaker/feature-store/) so that features can be reused and syndicated across ML initiatives and data science teams.

#### 2.1.1 Data Wrangler - Data Import

First, I imported the data from an Amazon [S3](https://aws.amazon.com/s3/) storage bucket where I had already uploaded the CSV file from the UCI ML repository using the following Python code.

```
# cell 00 ... setup Exploratory Data Analysis of Heart Disease Data
import boto3
import sagemaker
from sagemaker import get_execution_role

region = boto3.Session().region_name

session = sagemaker.Session()
s3_bucket = session.default_bucket()
s3_prefix = 'sagemaker/heartdisease/data/'

s3 = boto3.Session().resource('s3')
local_data_filename = 'heart_failure_clinical_records_data.csv'
!wget https://archive.ics.uci.edu/ml/machine-learning-databases/00519/heart_failure_clinical_records_dataset.csv -O heart_failure_clinical_records_data.csv

s3.Bucket(s3_bucket).upload_file(local_data_filename, s3_prefix + local_data_filename)
```

#### 2.1.2 Data Wrangler - Data Preparation - Data Types and Transformation Flow

Second, I applied data type transformations to convert the columns from the raw character/string types to numeric types because ML models typically require that the input columns be numeric.  

Third, I moved the target column to be the first column because that is expected for the built tabular data SageMaker algorithms such as Linear Learner, KNN, and XGBoost.  I also scaled some of the non-boolean numeric columns using the standardization transform so that the scaled columns have a mean of 0 and standard deviation of 1; these scaled columns reduce the impact of outliers and improve the fit of the algorithm to the supervised training data as well as its ability to  generalize, perfomring predictions on the evaluation and actual data in the real world.

For all of these columnar operations, Data Wrangler simplifies the data pipeline process with easy, built-in operations that can be created visually in a data pipeline [workflow](02a-DataWrangler-heartdisease-DataFlow.flow) without any code.

![Heart Disease Data Wrangler Data Flow](../assets/heart-disease-2.1.2-datawrangler-flow.png)

#### 2.1.3 Data Wrangler - Data Visualization

Fourth, I generated several visual, analytical outputs including correlations, histograms, and table summary statistics; again, all of these analytical visual artifacts are produced conveniently and simply within the visual IDE without any code.

![Heart Disease Correlation Heatmap](../assets/heart-disease-2.1.3-datawrangler-correlationheatmap.png)

Based on the correlation heatmap, one can see that the attributes for serum creatinine, ejection fraction, serum sodium, age, and blood pressure have the top 5 highest correlations with our primary target_heart_failure feature; moreover, there appears to be secondary correlation between sex and smoking, diabetes, and high blood pressure.  Furthermore, if one examines the histograms below for age, creatinine, and ejection fraction, one can observe high density clusters for each aforementioned attribute around 60+, above 1.0+, and below 40% respectively.

![Heart Disease Age Histogram](../assets/heart-disease-2.1.3-datawrangler-histogram-age.png)

![Heart Disease Creatinine Histogram](../assets/heart-disease-2.1.3-datawrangler-histogram-creatinine.png)

![Heart Disease Ejection Fraction Histogram](../assets/heart-disease-2.1.3-datawrangler-histogram-ejectionfraction.png)

#### 2.1.4 Data Wrangler - Data Analysis - Collinearity Lasso Feature Selection

Next, I performed a collinearity attribute analysis on the data set columns to double check the earlier correlations and ensure that the most important attributes are included in the ML model.  SageMaker Data Wrangler's Collinearity feature uses a simple Linear Model, and it can be run in the context of classification or regression problems.  In this case, it builds the linear classifier with L1 term regularization and you can control the strength of the L1 penalty.  The classifier itself has a ROC AUC score of 76.3% which is within the order of accuracy of previous work done on this data set without any tuning yet.  Furthermore, the collinearity analysis again confirms the importance age, ejection fraction, serum creatinine, sodium, and blood pressure.  Also, note that classifier standardizes features to have mean 0 and standard deviation 1.

![Heart Disease Feature Collinearity](../assets/heart-disease-2.1.4-datawrangler-feature-collinearity.png)

#### 2.1.5 Data Wrangler - Data Analysis - Quick Model

Next, I ran Data Wrangler's Quick Model analysis that uses the Random Forest algorithm to predict the binary classifaction of the target_heart_failure feature.  The model was trained with the default hyper parameters with no custom tuning and there was minimal preprocessing of the features.  Nonetheless, the model achieved an F1 score of 0.682 on the test data set and it also surfaces feature importance as a triple-check of earlier results, again confirming that ejection fraction, age, serum creatinine, and sodium are consequential features.

![Heart Disease Quick Model](../assets/heart-disease-2.1.5-datawrangler-quickmodel.png)

#### 2.1.6 Data Wrangler - Clean Up

When you are not using Data Wrangler, it is important to shut down the instance on which it runs to avoid incurring additional fees.

To avoid losing work, save your data flow before shutting Data Wrangler down. To save your data flow in Studio, select File and then select Save Data Wrangler Flow. Note that Data Wrangler automatically saves your data flow every 60 seconds.

There are a couple of choices:
- From SageMaker Studio, select the Running Instances and Kernels icon.  Under RUNNING APPS is the sagemaker-data-wrangler-1.0 app. Select the shut down icon next to this app.
- Install the [Jupyter Lab Extensions](https://aws.amazon.com/blogs/machine-learning/save-costs-by-automatically-shutting-down-idle-resources-within-amazon-sagemaker-studio/) for automatically detecting and shutting down idle resources such as a Data Wrangler.  

#### 2.2.1 Jupyter Notebook - Data Analysis

While much of the aformentioned data exploration can be performed conveniently in Data Wrangler, the inquisitive reader is invited to examine and inspect a Jupyter [notebook](02b-HeartDisease-ExploratoryDataAnalysis.ipynb) that performs several similar operations through the use of pandas, matplotlib, and seaborn Python libraries.  Again, this approach makes sense when your data set is very large (TBs of data with billions of rows), unstructured (e.g. image, audio, video, text), or requires complex column transformations beyond the existing set of 300+ built-in transformations available in Data Wrangler.

## :brain: 3.0 Heart Failure Machine Learning Model Build and Tune

Now that the data set has been engineered for features, one can begin to build, evaluate, and tune ML models using the robust, easy-to-use set of components available in Amazon SageMaker.

First, SageMaker supports seventeen (17) built-in [Algorithms](https://docs.aws.amazon.com/sagemaker/latest/dg/algos.html), and it also enables you to bring your own and customize them further Docker.  Since this Heart Disease project has been framed as a binary classification problem on tabular data, I planned to use the built-in Linear Learner, K-Nearest Neighbor, and XGBoost models which are all appropriate for tabular data.

Second, I planned to test different sets of hyperparameters across the aforementioned algorithms.  Managing all these iterative activities in an effective manner could be simplified using SageMaker [Experiments](https://aws.amazon.com/blogs/aws/amazon-sagemaker-experiments-organize-track-and-compare-your-machine-learning-trainings/) which enables you to organize, track, compare, and evaluate your machine learning experiments.  SageMaker Experiments automatically tracks the inputs, parameters, configurations, and results of your iterations as trials. You can assign, group, and organize these trials into experiments. SageMaker Experiments is integrated with Amazon SageMaker Studio providing a visual interface to browse your active and past experiments, compare trials on key performance metrics, and identify the best performing models.  In this project, there was one parent Experiment (sm-heart-exp-YYYY-mm-dd), one child trial for each algorithm (sm-heart-trial-algoZZZ-YYYY-mm-dd), and multiple trial components (e.g. grandchildren) representing a specific algorithm and a specific hyper parameter set.

Third, I needed to efficiently explore the universe of hyperparameters (HP) appropriate to each algorithm and find the best version of a model.  Again, this was an opportunity to leverage another SageMaker component for [Automated Model HyperParameter Tuning](https://aws.amazon.com/blogs/machine-learning/amazon-sagemaker-automatic-model-tuning-produces-better-models-faster/) which executes many training jobs in parallel on your data set using the algorithm and ranges of hyperparameters you specify.  Each HP job run is a trial component, and once the job set has completed, you can choose the best performing model as measured by a classification objective metric (e.g. AUC, Accuracy, F1, Logistic Loss) that you choose.

Fourth, I needed to understand what features were important to the best performing model in each algorithmic family and how important they were.  Amazon SageMaker [Clarify](https://aws.amazon.com/blogs/aws/new-amazon-sagemaker-clarify-detects-bias-and-increases-the-transparency-of-machine-learning-models/) provides tools to help explain how machine learning (ML) models make predictions as well as determine whether data used for training models encodes any bias. These tools can help ML modelers and developers and other internal stakeholders understand model characteristics as a whole prior to deployment and to debug predictions provided by the model after it's deployed. Transparency about how ML models arrive at their predictions is also critical to consumers and regulators who need to trust the model predictions if they are going to accept the decisions based on them. SageMaker Clarify uses a model-agnostic feature attribution implementation of [SHAP](https://docs.aws.amazon.com/sagemaker/latest/dg/clarify-shapley-values.html), based on the concept of a Shapley value from the field of cooperative game theory that assigns each feature an importance value for a particular prediction.

Each Jupyter Notebook for the different algorithms has several common code blocks for the environment setup as well as the ML model initialization, training, tuning, and evaluation.  Cell blocks 00-07 are parameterized by the algorithm ID, the S3 path for the model, splitting data into training (80%) and test (20%) data subsets, as well as the trial naming convention that is specific to the algorithm.

```
# cell 01
import sagemaker

import boto3
from sagemaker import get_execution_role

region = boto3.Session().region_name

# https://docs.aws.amazon.com/sagemaker/latest/dg/xgboost.html
# https://docs.aws.amazon.com/sagemaker/latest/dg/xgboost_hyperparameters.html
session = sagemaker.Session()
s3_bucket = session.default_bucket()
s3_data_prefix = 'sagemaker/heartdisease/data/'
s3_model_prefix = 'sagemaker/heartdisease/xgboost'
s3_output_path = 's3://{}/{}/output'.format(s3_bucket, s3_model_prefix)
algorithm = 'xgboost'
trial_prefix = 'sm-heart-xgb-trial'

role = get_execution_role()

sm = boto3.Session().client(service_name='sagemaker',region_name=region)
s3 = boto3.Session().resource('s3')
```

Experiments (grandparent) and trials (algorithmic parent) are constructed on a daily cadence to organize all the individual trial components (grandchildren) that represent specific ML models consisting of an algorithm and its related hyperparameters.

```
# cell 06
import time
from time import strftime

import smexperiments
from smexperiments.experiment import Experiment
from smexperiments.trial import Trial
from smexperiments.trial_component import TrialComponent
from smexperiments.tracker import Tracker

create_date = strftime("%Y-%m-%d")
experiment_prefix = 'sm-heart-exp' 
experiment_name = 'sm-heart-exp-{}'.format(create_date)
trial_name = '{}-{}'.format(trial_prefix, create_date)

# experiment
try:
    experiment = Experiment.load(experiment_name = experiment_name)
except Exception as ex:
    if "ResourceNotFound" in str(ex):
        experiment = Experiment.create(experiment_name = experiment_name, 
                                       description = "SageMaker Heart Disease experiment", 
                                       tags = [{'Key': 'Experiment', 'Value': experiment_name}])
# trial

try:
    trial = Trial.load(trial_name)
except Exception as ex:
    if "ResourceNotFound" in str(ex):
        trial = Trial.create(experiment_name=experiment_name, trial_name=trial_name)
        
```

Once the experiment and trial exist, one can construct an Estimator specific to the algorithm and initialize it with a debugger hook so that later on, one can later run Clarify bias and explainability processing jobs.

```
 cell 07
from sagemaker.debugger import rule_configs, Rule, DebuggerHookConfig, CollectionConfig

# get algo container [class]
container = sagemaker.image_uris.retrieve(region=boto3.Session().region_name, framework=algorithm, version='latest')
# setup estimator for algo [tbt running instance of algo class]
ml_model = sagemaker.estimator.Estimator(                
                container,
                role, 
                instance_count=1, 
                instance_type='ml.m4.xlarge',
                output_path=s3_output_path,
                sagemaker_session=session,
                tags = [{'Key':'Experiment','Value':experiment_name},{'Key':'Trial','Value':trial_name}],
                debugger_hook_config=DebuggerHookConfig(
                    s3_output_path = s3_output_path,
                    collection_configs = [
                        CollectionConfig(name="metrics",parameters={"save_interval":"5"}),
                        CollectionConfig(name="predictions",parameters={"save_interval":"5"}),
                        CollectionConfig(name="feature_importance",parameters={"save_interval":"5"}),
                        CollectionConfig(name="average_shap", parameters={"save_interval":"5"})
                    ]
                )                  
)

# setup training and test/validation channels
s3_input_train = sagemaker.inputs.TrainingInput(s3_data='s3://{}/{}/train'.format(s3_bucket, s3_model_prefix), content_type='csv')
s3_input_validation = sagemaker.inputs.TrainingInput(s3_data='s3://{}/{}/test/'.format(s3_bucket, s3_model_prefix), content_type='csv')
```

The code that is truly specific to the algorithm (parent trial) and its family of trial components occurs when defininig the HyperParameterTuner job and evaluate the model(s) once they have been tuned and trained.

```
#cell 08 ... HyperParameterTuner.Job.Run()

from sagemaker.tuner import IntegerParameter, CategoricalParameter, ContinuousParameter, HyperparameterTuner

ml_model.set_hyperparameters( num_round = 100)

# define hyperparameter ranges
# https://docs.aws.amazon.com/sagemaker/latest/dg/xgboost-tuning.html ... alpha, min_child_weight, eta, num_round
hyperparameter_ranges = {'eta': ContinuousParameter(0, 1),
                            'min_child_weight': ContinuousParameter(1, 10),
                            'alpha': ContinuousParameter(0, 100),
                            'max_depth': IntegerParameter(1, 10)}

# define metric ... F1 = harmonic mean of precision and recall
objective_metric_name = 'validation:auc'

# define HyperParameterTuner job
ml_tuner = HyperparameterTuner(ml_model,
                            objective_metric_name,
                            hyperparameter_ranges,
                            max_jobs=2,
                            max_parallel_jobs=2,
                            base_tuning_job_name = trial_prefix,
                           tags = [{'Key':'Experiment','Value':experiment_name},{'Key':'Trial','Value':trial_name}])

print('Defined ML model HyperParameterTuner job for {}'.format(trial_prefix))

# job.Run()
ml_tuner.fit({'train': s3_input_train, 'validation': s3_input_validation})
```

Once the HyperParameterTuner job has completed, one programmatically select the best performing ML model and then deploy.

```
# cell 10 ... select best training job per the eval metric
# HyperParameterTuner.model.select
ml_tuner.best_training_job()
# HyperParameterTuner.model.deploy
ml_tuner_predictor = ml_tuner.deploy(initial_instance_count=1, instance_type='ml.m4.xlarge', serializer=sagemaker.serializers.CSVSerializer())
```

Once the model has been deployed, one can evaluate and execute inference on the test data.

```
# cell 12 .. calculate predictions for XGBoost
import numpy as np
import json

def predict(predictor, data, rows=500, verbose=False):
    predictions = ''
    for row in data:
        if (verbose):
            print(row)
        prediction = predictor.predict(row, initial_args={"ContentType":"text/csv"})
        if (verbose):
            print(prediction)
        predictions = ','.join([predictions, prediction.decode('utf-8')])

    return np.fromstring(predictions[1:],sep=',')
```

```
 cell 13 ... analyze predictions on 1 record and then for the whole test data set

result = ml_tuner_predictor.predict([1,0,0,0,0,-0.491279276,7.514639529,-0.00707675,7.54E-09,-0.284552352,-0.141976151], initial_args={"ContentType":"text/csv"})
print(result)

# print (test_features)
predictions = predict(ml_tuner_predictor, test_features.to_numpy(), verbose=False)

print(predictions)

# calculate confusion matrix
pd.crosstab(index=test_labels, columns=np.round(predictions), rownames=["actual"],colnames=["predictions"])
```
Once there are some predictions on test data, we can analyze the confusion matrix and calculate the important classification metrics (e.g. Precision, Recall, Accuracy, F1, AUC) using the scikit-learn metrics module.  Note in block 12 of the notebooks that the predict helper function slightly differs based on the algorithm's response output format.

```
# cell 14 ... calculate classification metrics

from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score
print('Precision = {}'.format(precision_score(test_labels, predictions, average='weighted', zero_division=1)))
print('Recall = {}'.format(recall_score(test_labels, predictions, average='weighted', zero_division=1)))
print('Accuracy = {}'.format(accuracy_score(test_labels, predictions)))
print('F1 = {}'.format(f1_score(test_labels, predictions, average='weighted', zero_division=1)))
print('AUC = {}'.format(roc_auc_score(test_labels, predictions, average='weighted')))
```

#### 3.1 Linear Learner Model

Amazon SageMaker Linear Learner (LL) is an algorithm that learns a linear function and maps an input vector X of dependent variables to an approximation of the independent variable Y.  LL is appropriate for both classification and regression problems.  The LL function is trained with a distributed implementation of Stochastic Gradient Descent (SGD).  LL requires a tabular data matrix with rows representing observations and columns representing the dimensions of the features.  It also requires a column that contains the labels that match the data points.

The first algorithm specific code block is for S3 and algorithm initialization in code block 01.  The algorithm variable is later used to identify the proper SageMaker Docker container when constructing the Estimator object in code block 08.

```
# cell 01
import sagemaker

import boto3
from sagemaker import get_execution_role

region = boto3.Session().region_name

# https://sagemaker.readthedocs.io/en/v1.11.0/linear_learner.html
# https://docs.aws.amazon.com/sagemaker/latest/dg/linear-learner-tuning.html
# https://docs.aws.amazon.com/sagemaker/latest/dg/ll_hyperparameters.html
session = sagemaker.Session()
s3_bucket = session.default_bucket()
s3_data_prefix = 'sagemaker/heartdisease/data/'
s3_model_prefix = 'sagemaker/heartdisease/linearlearner'
s3_output_path = 's3://{}/{}/output'.format(s3_bucket, s3_model_prefix)
algorithm = 'linear-learner'
trial_prefix = 'sm-heart-ll-trial'
```

The second algorithm specific code block is for HyperParameter specification in cell 08.

```
#cell 08 ... HyperParameterTuner.Job.Run()

from sagemaker.tuner import IntegerParameter, CategoricalParameter, ContinuousParameter, HyperparameterTuner

# base hyper parameters ... binary classifier with optimizer using Stochastic Gradient Descent (SGD)
ml_model.set_hyperparameters( num_classes=2, predictor_type='binary_classifier', loss='auto', epochs=10, optimizer='sgd',mini_batch_size=100)

# define hyperparameter ranges
# https://docs.aws.amazon.com/sagemaker/latest/dg/linear-learner-tuning.html ... L1 penalty term, learning rate
hyperparameter_ranges = {
                            'l1': ContinuousParameter(0.1, 1),
                            'learning_rate': ContinuousParameter(0.1, 1)
                        }

# define metric ... F1 = harmonic mean of precision and recall
objective_metric_name = 'validation:binary_f_beta'
```

The third and last algorithm specific code section is the predictor helper function in cell 12.

```
# cell 12 .. calculate predictions for Linear Learner
import numpy as np
import json

def predict(predictor, data, verbose=False):
    predictions = []
    for row in data:
        if (verbose):
            print(row)
        prediction = json.loads(predictor.predict(row, initial_args={"ContentType":"text/csv"}))
        if (verbose):
            print(prediction)
        predictions += [r["predicted_label"] for r in prediction["predictions"]]
        
    return predictions
```


#### 3.2 K-Nearest Neighbor Model

Amazon SageMaker K-Nearest Neighbor (KNN) is a non-parametric, supervised algorithm that queries the k points that are closes to the sample point and returns the most frequently used label of their class or average of the independent variable as the prediction.  KNN is appropriate for both classification and regression problems.  KNN requires a tabular data matrix with rows representing observations and columns representing the dimensions of the features.  It also requires a column that contains the labels that match the data points.    During training, KNN is constructing an index that enables efficient lookups of distances betwen points whose values or class labels have not yet been determined and the k nearest points to use for inference.

The first algorithm specific code block is for S3 and algorithm initialization in cell 01.  The algorithm variable used to identify the proper SageMaker Docker container when constructing the Estimator object in block cell 08.

```
# cell 01
import sagemaker

import boto3
from sagemaker import get_execution_role

region = boto3.Session().region_name

# https://docs.aws.amazon.com/sagemaker/latest/dg/k-nearest-neighbors.html
# https://docs.aws.amazon.com/sagemaker/latest/dg/kNN_hyperparameters.html
session = sagemaker.Session()
s3_bucket = session.default_bucket()
s3_data_prefix = 'sagemaker/heartdisease/data/'
s3_model_prefix = 'sagemaker/heartdisease/knn'
s3_output_path = 's3://{}/{}/output'.format(s3_bucket, s3_model_prefix)
algorithm = 'knn'
trial_prefix = 'sm-heart-knn-trial'
```

The second algorithm specific code block is for HyperParameter specification in cell 08.

```
#cell 08 ... HyperParameterTuner.Job.Run()

from sagemaker.tuner import IntegerParameter, CategoricalParameter, ContinuousParameter, HyperparameterTuner

# base hyper parameters ... binary classifier with optimizer using Stochastic Gradient Descent (SGD)
ml_model.set_hyperparameters( predictor_type='classifier', feature_dim=11, sample_size=100)

# define hyperparameter ranges
# https://docs.aws.amazon.com/sagemaker/latest/dg/kNN_hyperparameters.html ... K <= SQRT(N), SQRT(300) = 17
hyperparameter_ranges = {
                            'k': IntegerParameter(1, 32),
                            'sample_size': IntegerParameter(256, 299)
                        }

# define metric ... Accuracy = (TP+TN) / (TP+TN+FP+FN)
objective_metric_name = 'test:accuracy'
```

The third and last algorithm specific code section is the predictor helper function in cell 12.

```
# cell 12 .. calculate predictions for KNN
import numpy as np
import json

def predict(predictor, data, verbose=False):
    predictions = []
    for row in data:
        if (verbose):
            print(row)
        prediction = json.loads(predictor.predict(row, initial_args={"ContentType":"text/csv"}))
        if (verbose):
            print(prediction)
        predictions += [r["predicted_label"] for r in prediction["predictions"]]
        
    return predictions
```

#### 3.3 XGBoost Model

Amazon SageMaker XGBoost (XGB) is a supervised algorithm that uses an ensemble of estimates from a set of simpler and weaker tree models that can make predictions for classification and regression problems.  Each sub-tree maps an input data point to one of its leafs that contains the prediction (either a numeric value for regression or label for classification); XGB then minimizes an objective function that combines loss/error and a penalty term for model complexity.  XGB requires a tabular data matrix with rows representing observations and columns representing the dimensions of the features.  It also requires a column that contains the labels that match the data points.  XGB has grown in popularity due to its ability generalize and handle 

The first algorithm specific code block is for S3 and algorithm initialization in cell 01.  The algorithm variable used to identify the proper SageMaker Docker container when constructing the Estimator object in block cell 08.

```
# cell 01
import sagemaker

import boto3
from sagemaker import get_execution_role

region = boto3.Session().region_name

# https://docs.aws.amazon.com/sagemaker/latest/dg/xgboost.html
# https://docs.aws.amazon.com/sagemaker/latest/dg/xgboost_hyperparameters.html
session = sagemaker.Session()
s3_bucket = session.default_bucket()
s3_data_prefix = 'sagemaker/heartdisease/data/'
s3_model_prefix = 'sagemaker/heartdisease/xgboost'
s3_output_path = 's3://{}/{}/output'.format(s3_bucket, s3_model_prefix)
algorithm = 'xgboost'
trial_prefix = 'sm-heart-xgb-trial'
```

The second algorithm specific code block is for HyperParameter specification in cell 08.

```
from sagemaker.tuner import IntegerParameter, CategoricalParameter, ContinuousParameter, HyperparameterTuner

ml_model.set_hyperparameters( num_round = 100)

# define hyperparameter ranges
# https://docs.aws.amazon.com/sagemaker/latest/dg/xgboost-tuning.html ... alpha, min_child_weight, eta, num_round
hyperparameter_ranges = {'eta': ContinuousParameter(0, 1),
                            'min_child_weight': ContinuousParameter(1, 10),
                            'alpha': ContinuousParameter(0, 100),
                            'max_depth': IntegerParameter(1, 10)}

# define metric ... F1 = harmonic mean of precision and recall
objective_metric_name = 'validation:auc'

# define HyperParameterTuner job
ml_tuner = HyperparameterTuner(ml_model,
                            objective_metric_name,
                            hyperparameter_ranges,
                            max_jobs=100,
                            max_parallel_jobs=5,
                            base_tuning_job_name = trial_prefix,
                           tags = [{'Key':'Experiment','Value':experiment_name},{'Key':'Trial','Value':trial_name}])

```

The third and last algorithm specific code section is the predictor helper function in cell 12.

```
# cell 12 .. calculate predictions for XGBoost
import numpy as np
import json

def predict(predictor, data, rows=500, verbose=False):
    predictions = ''
    for row in data:
        if (verbose):
            print(row)
        prediction = predictor.predict(row, initial_args={"ContentType":"text/csv"})
        if (verbose):
            print(prediction)
        predictions = ','.join([predictions, prediction.decode('utf-8')])

    return np.fromstring(predictions[1:],sep=',')
```

#### 3.4 Comparing the ML Models

Each Jupyter Notebook for each algorithm shows how to assess the best performing model within the algorithmic family of trial components by running a test data set against an inference endpoint hosting the tuned model.  Classification metrics were then calculated using the scikit-learn metrics module.  The XGBoost model has best overall performance The KNN model had a F1 score of 71%, the XGBoost model had a F1 score of 74%, and the Linear Learner model had a F1 score of 83%.  These results compare favorably with earlier work by Chicco and Jurman referenced earlier.

SageMaker Experiments also makes it simple to visually and numerically compare the model results spanning hundreds to thousands of individual trial components that comprise different algorithms and different hyperparameters all organized as part of one larger experiment associated to a business use case.  Note under the grandparent experiment (sm-heart-exp-2021-12-08), there are three algorithmic parent trials (sm-heart-ll-trial-2021-12-08, sm-heart-xgb-trial-2021-12-08, sm-heart-knn-trial-2021-12-08) each associated to HyperParameterTuner job consisting of a set of trial component grandchildren runs corresponding to individual training jobs.  Again, the virtue of SageMaker is that it provides a flexible Experiment framework so that we can expand this experiment by widening the hyperparameter ranges, increasing the number of training jobs, and by even introducing additional algorithms such as Artificial Neural Nets without fundamentally altering the Notebook code or Experiment Management; it becomes a matter of gently amending the initialization variables and the hyperparameter ranges specific to the algorithm.

![Heart Disease Model Experiments](../assets/heart-disease-3.4-experiments.png)

## :computer: 4.0 Heart Failure Machine Learning Model Deployment

abc123

## :sun_behind_large_cloud: 5.0 Conclusion

abc123
