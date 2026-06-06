from pandas import DataFrame
from joblib import load
from pydantic import BaseModel, ValidationError
from fastapi import FastAPI, HTTPException

model = load("salary_pipeline.pkl")

app = FastAPI(
    title="Data Science Salary Predictor",
    description="Predice el salario anual en USD de un profesional de Data Science.",
    version="1.0.0",
)


class DataPredict(BaseModel):
    data_to_predict: list[list] = [
        [2023, "SE", "FT", "Data Scientist", 100, "US", "US", "M"],
        [2023, "EN", "FT", "Data Analyst",     0, "IN", "IN", "S"],
    ]


class PredictResponse(BaseModel):
    prediction: list[float]


@app.post("/predict", response_model=PredictResponse)
def predict(request: DataPredict):
    try:
        df_data = DataFrame(
            request.data_to_predict,
            columns=[
                "work_year", "experience_level", "employment_type",
                "job_title", "remote_ratio", "company_location",
                "employee_residence", "company_size",
            ],
        )
        prediction = model.predict(df_data)
        return {"prediction": [round(p, 2) for p in prediction.tolist()]}
    except ValidationError as ve:
        raise HTTPException(status_code=400, detail=ve.errors())
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
def home():
    return {"Universidad EIA": "MLOps - Taller 3"}
