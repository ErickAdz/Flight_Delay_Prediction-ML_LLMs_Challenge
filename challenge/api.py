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
    return {"message": "Welcome to Delay Model API"}

@app.get("/health", status_code=200)
async def get_health() -> dict:
    return {
        "status": "OK"
    }

@app.post("/predict")
async def post_predict(data: dict) -> dict:
    """
    Endpoint for making predictions using the DelayModel.
    Expects a JSON request with input data.
    """
    try:
        # Convert the "flights" list from the input data into the required format
        flights = data.get("flights", [])

        # Validation for "TIPOVUELO" and "MES" in each flight
        for flight in flights:
            tipo_vuelo = flight.get("TIPOVUELO")
            month = flight.get("MES", 0)

            # Check "TIPOVUELO"
            if tipo_vuelo not in ("N", "I"):
                raise HTTPException(status_code=400, detail="Invalid 'TIPOVUELO' value. Must be 'N' or 'I'.")

            # Check "MONTH"
            if not (1 <= month <= 12):
                raise HTTPException(status_code=400, detail="Invalid 'MONTH' value. Must be an integer from 1 to 12.")
        
        # Create a DataFrame with the input data for the predict method
        features = pd.DataFrame(flights)

        features = delay_model.preprocess(features)
        predictions = delay_model.predict(features)

        # Prepare the response
        response = {"predict": predictions}

        return response

    except HTTPException as http_exc:
        # Reraise the HTTPException without modifying it
        raise http_exc
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")