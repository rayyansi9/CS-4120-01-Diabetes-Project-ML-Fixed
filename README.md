# CS-4120-01-Diabetes-Project-ML
Set up and running instructions

#Entire code is first being fixed on the deefback from midpoint report and then we're implement a NN.

This project predicts diabetes disease progression using the Diabetes Dataset, which is publicly available in scikit-learn. datasets. We train regression models to predict the continuos progression score and add a media n-based label to build classification baselines.

To run the project, first clone our repository and open the folder in an appropriate IDE/terminal. Install all required packages, which are specified with the versions we used in requirements.txt. You don’t need to download the dataset manually; it loads automatically through src/data.py.

After loading the data, you can train our baseline models by running python src/train_baselines.py. This will train both Linear Regression and Decision Tree Regressor for the regression task, as well as Logistic Regression and Decision Tree Classifier for the classification task. The results will be saved in the form of CSV tables under src/notebooks/tables/.

If you want to generate visualization plots, which are the correlation heatmap, confusion matrix, residual plots, and target distribution, run python src/evaluationplots.py. The plots will then generate and stored automatically in src/notebooks/figures/.

Our best models for predicting numerical values were Linear Regression, and for classifying data, we used Logistic Regression. Both models showed consistent results during testing, suggesting that the data is mostly linear and well-structured. These models provide a strong starting point before we move on to testing deeper neural network designs and using optimizers like Adam to improve performance.

Quick Results Summary
- Linear Regression performed best for regression (lowest MSE).
- Logistic Regression achieved the highest accuracy for classification.
- Residuals showed a roughly normal distribution, indicating a stable model fit.
- The dataset was balanced, and both models generalized reasonably well.

Some Important Points to Note
- Random seeds are fixed throughout (random_state=42) for reproducibility.
- No raw data is committed — the dataset loads automatically from scikit-learn.
- The project structure is organized to be compatible with MLflow for later logging, though tracking is not yet active in this phase.

We used ChatGPT & Gemini Pro for code review and formatting guidance for the plots, specifically for implementing the color scheme. Program structures are inspired by all the Lecture Codes uploaded on Moodle.
