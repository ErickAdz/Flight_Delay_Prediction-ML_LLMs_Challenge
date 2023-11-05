import fastapi
import pandas as pd
from challenge.model import DelayModel
from fastapi import HTTPException, Request, responses

delay_model = DelayModel()

app = fastapi.FastAPI()

app = fastapi.FastAPI(title="Delay Model API", openapi_url="/openapi.json")

@app.get("/", status_code=200)
async def root():
    """
    Root GET
    """
    return responses.FileResponse("../index.html") 

@app.get("/health", status_code=200)
async def get_health() -> dict:
    return {
        "status": "OK"
    }

@app.post("/predict", status_code=200)
async def post_predict(data: dict) -> dict:
    """
    Endpoint for making predictions using the DelayModel.
    Expects a JSON request with input data.
    """
    try:
        # Convert the "flights" list from the input data into the required format
        flights = data.get("flights", [])

        # Create a DataFrame with the input data for the predict method
        features = pd.DataFrame(flights)

        # Preprocess the DataFrame to handle object columns
        # You may need to one-hot encode or convert categorical columns
        features = delay_model.preprocess(features)

        print("at least the problem is not here, wich is worse thinkin about it .-.  ", features)

        # Call the predict method on the DelayModel instance
        predictions = delay_model.predict(features)

        # Prepare the response
        response = {"predictions": predictions}

        return response
    except Exception as e:
        # Handle any exceptions or errors
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")