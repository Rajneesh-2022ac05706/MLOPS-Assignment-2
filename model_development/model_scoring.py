import joblib
import pickle
import numpy as np


xgboost_model = joblib.load('../model_artifacts/xgboost_v1.joblib')

with open('../model_artifacts/categorical_column_unique_values.pkl', 'rb') as f:
    categorical_column_unique_values = pickle.load(f)
label_encoders = {"workclass":joblib.load("../model_artifacts/label_encoder_workclass.joblib"),
                  "education_num": joblib.load("../model_artifacts/label_encoder_education_num.joblib"),
                  "marital_status":joblib.load("../model_artifacts/label_encoder_marital_status.joblib"),
                  "occupation":joblib.load("../model_artifacts/label_encoder_occupation.joblib"), 
                  "relationship":joblib.load("../model_artifacts/label_encoder_relationship.joblib"),
                  "asset_code":joblib.load("../model_artifacts/label_encoder_asset_code.joblib")}


def process_input_data(json_list):
    """
    Processes a list of JSON objects to apply label encoding and prepare for prediction.
    """
    processed_data = []
    for input_data in json_list:
        prepared_data = []
        for feature in xgboost_model.get_booster().feature_names:
            if feature in ["capital_gain_is_zero","capital_loss_is_Zero"]:
                feature=feature.replace("_is_zero","").replace("_is_Zero","")
            else:
                feature = feature.replace("_label_encoded","")

            if feature not in input_data:
                raise AssertionError(f"Expected feature {feature}, but it was not provided.")
            if feature in label_encoders:
                encoder = label_encoders[feature]
                if feature in label_encoders: 
                    if input_data[feature] not in categorical_column_unique_values[feature]:
                        raise AssertionError(f"The given feature value does not match with expected values for feat: {feature}")
                prepared_data.append(encoder.transform([input_data[feature]])[0])
            elif feature in ["capital_gain","capital_loss"]:
                val = 1 if input_data[feature] > 0 else 0 
                prepared_data.append(val)
            else:
                prepared_data.append(input_data[feature])
        processed_data.append(prepared_data)
    return np.array(processed_data)

def predict(json_list):
    """
    Predicts the output for a batch of input JSON objects using the loaded XGBoost model.
    """
    processed_data = process_input_data(json_list)
    predictions = xgboost_model.predict_proba(processed_data)
    return [i[1] for i in predictions]

json_list = [
    {
        'capital_gain': 0,
        'capital_loss': 0,
        'workclass': ' Private',
        'education_num': 16,
        'marital_status': ' Married-civ-spouse',
        'occupation': ' Exec-managerial',
        'relationship': ' Husband',
        'asset_code': 1,
        'capital_profit': 0,
        'age_of_applicant': 60
    },
    {
        'capital_gain': 7000,
        'capital_loss': 20,
        'workclass': ' Local-gov',
        'education_num': 7,
        'marital_status': ' Never-married',
        'occupation': ' Transport-moving',
        'relationship': ' Unmarried',
        'asset_code': 2,
        'capital_profit': 7000,
        'age_of_applicant': 28
    }
]

predictions = predict(json_list)
print(predictions)