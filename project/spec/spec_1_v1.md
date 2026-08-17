# feature: concept creation

## context
An application for non-technical learners to apply supervised machine learning techniques. Where they can load data, select the model and train it. After that they can enter new data to measure the model's result.

## requirements
- Must be developed using python, stremlit and sckit-learn;
- Should allow uploading xlsx, csv, txt and json files. Must identify the separator automatically;
- Must display the loaded table for screen viewing;
- Must allow selection between models for classification and regression models;
- Must list at least 4 models of each type;
- Must show a loading during training;
- Must show the metrics achieved by the model;
- It must allow the creation of inputs based on the loaded data, for model selection;
- It must show the level of reliability of the model in the prediction made (whether classifying or regression).

## architectural requirements
- It should be modular, separating the backend and frontend. The backend will have modules for each part of the system.

## front-end requirements
- It must happen in stages:
    1 - Load the file and view the table;
    2 - Select the target, application type (calissification or regression), model (with the full name of the model) - Place a 2-line explanation over each model when it is selected
    3 - Training result (metrics)
    4 - Screen for measurement, creates inputs based on the uploaded file.
    5 - result of the measurement.
- Must contain two screens:
    1 - For data loading and training
    2 - For model affection
- Must contain clear titles and a fluid, step-by-step UI.