import pandas as pd
from joblib import load
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
import warnings

warnings.filterwarnings("ignore", message="X does not have valid feature names")

loaded_knn = load('models/knn_tuned.joblib')
loaded_scaler = load('models/scaler_obj.joblib')
loaded_encoder = load('models/cat_encoder.joblib')

# Sample input
# input_data = ["Sunday", "Air India", "Business", "Delhi",	"6 AM - 12 PM",	"1-stop", "After 6 PM",	"Ahmedabad", 9.9167, 1]
input_data = ["Wednesday","Vistara","Business","Hyderabad","12 PM - 6 PM","1-stop","Before 6 AM","Delhi",13,40]

def knn_predict(data, scaler_obj:StandardScaler, model:KNeighborsRegressor, encoder:OneHotEncoder):
    """Predicts the price using a knn regressor model

    Parameters:
        data: Pandas Dataframe,
        scaler_obj: Loaded standard scaler,
        model: Loaded knn model (or some other trained model),
        encoder: Fitted encoder, to encode the categorical string columns

    Returns:
        prediction: It returns the predicted price of the flight fare
    """
    input_df = pd.DataFrame(data=[data])
    num_input_df = input_df.select_dtypes(include="number")
    obj_input_df = input_df.select_dtypes(include="object")

    scaled_array = scaler_obj.transform(num_input_df)
    scaled_num_df = pd.DataFrame(data=scaled_array)
    encoded_obj_arr = encoder.transform(obj_input_df).toarray()
    encoded_obj_df = pd.DataFrame(data=encoded_obj_arr)

    final_encoded_df = pd.concat(objs=[scaled_num_df, encoded_obj_df], axis=1)
    prediction = model.predict(final_encoded_df)

    return prediction[0]

if __name__ == "__main__":
    pred_value = knn_predict(
        data=input_data,
        scaler_obj=loaded_scaler,
        model=loaded_knn,
        encoder=loaded_encoder
    )
    print(f"Prediction: {round(pred_value,2)} INR")




    