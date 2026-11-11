import os
from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel
from supabase import create_client, Client
from app.ml_model import ProductivityPredictor

# Safely load Supabase credentials from environment variables
SUPABASE_URL = os.environ.get("SUPABASE_URL", "").strip()
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "").strip()

supabase: Client | None = None
if SUPABASE_URL and SUPABASE_KEY:
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

app = FastAPI(title="Smart Screen-Time & Productivity Predictor")
predictor = ProductivityPredictor()

class UsageInput(BaseModel):
    social_media_mins: int
    study_app_mins: int
    hour_of_day: int

@app.get("/")
def home():
    return FileResponse("static/index.html")

@app.post("/predict")
def get_prediction(data: UsageInput):
    result = predictor.predict(
        data.social_media_mins, 
        data.study_app_mins, 
        data.hour_of_day
    )
    
    if supabase:
        supabase.table("usage_logs").insert({
            "social_media_mins": data.social_media_mins,
            "study_app_mins": data.study_app_mins,
            "hour_of_day": data.hour_of_day,
            "prediction": result
        }).execute()

    return {
        "input_summary": data,
        "ai_prediction": result
    }

@app.get("/history")
def get_history():
    if not supabase:
        return []
    response = supabase.table("usage_logs").select("*").order("id", desc=True).limit(5).execute()
    return response.data
