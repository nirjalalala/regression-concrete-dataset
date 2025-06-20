# regression-concrete-dataset
Two models were trained and tested using dataset in
https://www.kaggle.com/datasets/elikplim/concrete-compressive-strength-data-set/data .

The baseline model (Random Forest Regressor) was selected after comparing metrics and errors from five different regression models.
The tuned model is a result of tuning the hyperparameters of baseline model with RandomSearchCV.

Finally, the web-app was designed and deployed using Streamlit.

Try the hosted app on - https://regression-concrete-dataset.streamlit.app
