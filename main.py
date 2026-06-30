import os
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, Request, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from jinja2 import Environment, FileSystemLoader, TemplateNotFound
import g4f
import asyncio

app = FastAPI()

# ---------- Path Setup ----------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
STATIC_DIR = os.path.join(BASE_DIR, "static")
DATA_FILE = os.path.join(BASE_DIR, "Soil_Nutrients_EnhancedING finaly.csv")

os.makedirs(TEMPLATES_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# ---------- Custom Template Renderer ----------
jinja_env = Environment(
    loader=FileSystemLoader(TEMPLATES_DIR),
    auto_reload=True,
    enable_async=False
)

def render_template(template_name: str, context: dict) -> HTMLResponse:
    try:
        template = jinja_env.get_template(template_name)
        content = template.render(**context)
        return HTMLResponse(content=content)
    except TemplateNotFound:
        return HTMLResponse(content=f"<h1>Template '{template_name}' not found</h1>", status_code=404)
    except Exception as e:
        return HTMLResponse(content=f"<h1>Template error: {e}</h1>", status_code=500)

# ---------- Load Model and Artifacts ----------
ARTIFACTS_FILE = os.path.join(BASE_DIR, "soil_model_xgb.pkl")
artifacts = None
try:
    artifacts = joblib.load(ARTIFACTS_FILE)
    model = artifacts['model']
    scaler = artifacts['scaler']
    target_encoder = artifacts['target_encoder']
    # New: OrdinalEncoder for all categorical columns
    feature_encoder = artifacts['feature_encoder']
    feature_names = artifacts['feature_names']
    # Define categorical columns (must match training order)
    categorical_columns = ['Fertility', 'Photoperiod', 'Category_pH', 'Soil_Type', 'Season']
    print("✅ Model and artifacts loaded successfully from soil_model_xgb.pkl")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    print("   Please run train_xgboost_model.py first to generate the file.")

# ---------- Compute Crop Summary Statistics ----------
crop_stats = {}
if os.path.exists(DATA_FILE):
    try:
        df = pd.read_csv(DATA_FILE)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = categorical_columns

        for crop in df['Name'].unique():
            crop_data = df[df['Name'] == crop]
            stats = {}
            for col in numeric_cols:
                stats[col] = {
                    'min': round(crop_data[col].min(), 2),
                    'max': round(crop_data[col].max(), 2),
                    'mean': round(crop_data[col].mean(), 2)
                }
            for col in cat_cols:
                mode_val = crop_data[col].mode().iloc[0] if not crop_data[col].mode().empty else "N/A"
                stats[col] = {'mode': mode_val}
            crop_stats[crop] = stats
        print(f"✅ Loaded crop statistics for {len(crop_stats)} crops")
    except Exception as e:
        print(f"❌ Error computing crop stats: {e}")
else:
    print(f"⚠️ Data file not found at {DATA_FILE}, crop stats will be empty.")

# ---------- Helper: Preprocess Input ----------
def preprocess_input(data_dict):
    df_input = pd.DataFrame([data_dict])
    # Encode categorical columns using the OrdinalEncoder
    cat_data = df_input[categorical_columns].values
    encoded_cats = feature_encoder.transform(cat_data)
    df_input[categorical_columns] = encoded_cats
    # Ensure numeric columns are float
    numeric_cols = [col for col in feature_names if col not in categorical_columns]
    for col in numeric_cols:
        df_input[col] = df_input[col].astype(float)
    # Reorder columns to match training order
    df_input = df_input[feature_names]
    # Scale all features
    scaled = scaler.transform(df_input)
    return scaled

# ---------- Chatbot Endpoint ----------
async def get_ai_reply(user_message: str) -> str:
    try:
        response = await asyncio.wait_for(
            g4f.ChatCompletion.create_async(
                model=g4f.models.default,
                messages=[
                    {"role": "system", "content": (
                        "You are a professional agricultural expert. "
                        "Reply in the language the user uses. "
                        "Ask 2-3 follow-up questions about soil, climate, crops. "
                        "Politely reject non-agricultural topics."
                    )},
                    {"role": "user", "content": user_message}
                ],
            ),
            timeout=30.0
        )
        return response
    except asyncio.TimeoutError:
        return "⏳ The AI took too long to respond. Please try again."
    except Exception as e:
        print(f"g4f error: {e}")
        return "🌱 Sorry, I'm having trouble connecting. Please try again later."

@app.post("/chat")
async def chat_endpoint(request: Request):
    data = await request.json()
    user_msg = data.get("message", "").strip()
    if not user_msg:
        return {"reply": "Please write something 😊"}
    reply = await get_ai_reply(user_msg)
    return {"reply": reply}

# ---------- API Endpoint for Crop Statistics ----------
@app.get("/api/crop_stats")
async def get_crop_stats():
    fallback_crops = list(target_encoder.classes_) if artifacts else []
    return {
        "crop_stats": crop_stats,
        "fallback_crops": fallback_crops
    }

# ---------- Routes ----------
@app.get("/", response_class=HTMLResponse)
async def welcome(request: Request):
    return render_template("welcome.html", {"request": request})

@app.get("/dashboard.html", response_class=HTMLResponse)
async def dashboard(request: Request):
    return render_template("dashboard.html", {"request": request})

@app.get("/soil.html", response_class=HTMLResponse)
async def soil(request: Request):
    return render_template("soil.html", {"request": request})

@app.get("/crop.html", response_class=HTMLResponse)
async def crop(request: Request):
    return render_template("crop.html", {
        "request": request,
        "stats": {
            "total_samples": len(target_encoder.classes_) if artifacts else 0,
            "accuracy": "97.5%",
            "model_type": "XGBoost"
        }
    })

@app.get("/map.html", response_class=HTMLResponse)
async def map_page(request: Request):
    return render_template("map.html", {"request": request})

@app.post("/predict_soil")
@app.post("/soil.html")
async def predict_soil(
    request: Request,
    Fertility: str = Form("Low"), Photoperiod: str = Form("Day Neutral"),
    Temperature: float = Form(0), Rainfall: float = Form(0), pH: float = Form(0),
    Light_Hours: float = Form(0), Light_Intensity: float = Form(0), Rh: float = Form(0),
    Nitrogen: float = Form(0), Phosphorus: float = Form(0), Potassium: float = Form(0),
    Category_pH: str = Form("neutral"), Soil_Type: str = Form("Loam"), Season: str = Form("Spring")
):
    # Input Validation
    errors = []
    if Temperature < -10 or Temperature > 60:
        errors.append(f"Temperature ({Temperature}°C) is out of realistic range (-10 to 60°C).")
    if Rainfall < 0 or Rainfall > 5000:
        errors.append(f"Rainfall ({Rainfall} mm) is out of realistic range (0-5000 mm).")
    if pH < 0 or pH > 14:
        errors.append(f"pH ({pH}) is out of range (0-14).")
    if Light_Hours < 0 or Light_Hours > 24:
        errors.append(f"Light Hours ({Light_Hours}) must be between 0 and 24.")
    if Light_Intensity < 0 or Light_Intensity > 200000:
        errors.append(f"Light Intensity ({Light_Intensity}) is out of realistic range (0-200000 lux).")
    if Rh < 0 or Rh > 100:
        errors.append(f"Relative Humidity ({Rh}) must be between 0 and 100%.")
    if Nitrogen < 0 or Nitrogen > 500:
        errors.append(f"Nitrogen ({Nitrogen}) is out of typical range (0-500 ppm).")
    if Phosphorus < 0 or Phosphorus > 500:
        errors.append(f"Phosphorus ({Phosphorus}) is out of typical range (0-500 ppm).")
    if Potassium < 0 or Potassium > 500:
        errors.append(f"Potassium ({Potassium}) is out of typical range (0-500 ppm).")
    
    if errors:
        error_msg = "Invalid input: " + " ".join(errors)
        if request.headers.get("X-Requested-With") == "XMLHttpRequest":
            return {"error": error_msg}
        return render_template("soil.html", {"request": request, "error": error_msg})
    
    # Prediction
    try:
        if artifacts is None:
            if request.headers.get("X-Requested-With") == "XMLHttpRequest":
                return {"error": "Model not loaded. Please train the model first."}
            return render_template("soil.html", {"request": request, "error": "Model not loaded."})

        input_dict = {
            'Fertility': Fertility,
            'Photoperiod': Photoperiod,
            'Temperature': Temperature,
            'Rainfall': Rainfall,
            'pH': pH,
            'Light_Hours': Light_Hours,
            'Light_Intensity': Light_Intensity,
            'Rh': Rh,
            'Nitrogen': Nitrogen,
            'Phosphorus': Phosphorus,
            'Potassium': Potassium,
            'Category_pH': Category_pH,
            'Soil_Type': Soil_Type,
            'Season': Season
        }

        X_input = preprocess_input(input_dict)
        probs = model.predict_proba(X_input)[0]
        pred_idx = np.argmax(probs)
        confidence = float(probs[pred_idx] * 100)
        
        CONFIDENCE_THRESHOLD = 30.0
        if confidence < CONFIDENCE_THRESHOLD:
            message = f"⚠️ No suitable crop found for these conditions. The highest confidence was {confidence:.1f}%, below the required {CONFIDENCE_THRESHOLD}%."
            if request.headers.get("X-Requested-With") == "XMLHttpRequest":
                return {"error": message}
            return render_template("soil.html", {"request": request, "error": message})
        
        predicted_crop = target_encoder.inverse_transform([pred_idx])[0]
        
        # Generate recommendations
        recommendations = []
        if predicted_crop in crop_stats:
            stats = crop_stats[predicted_crop]
            numeric_fields = ['Temperature', 'Rainfall', 'pH', 'Light_Hours', 'Light_Intensity', 'Rh', 'Nitrogen', 'Phosphorus', 'Potassium']
            for field in numeric_fields:
                if field in stats:
                    val = input_dict[field]
                    min_val = stats[field]['min']
                    max_val = stats[field]['max']
                    if val < min_val:
                        recommendations.append(f"{field} is too low ({val}). Recommended range for {predicted_crop}: {min_val} – {max_val}.")
                    elif val > max_val:
                        recommendations.append(f"{field} is too high ({val}). Recommended range for {predicted_crop}: {min_val} – {max_val}.")
            cat_fields = categorical_columns
            for field in cat_fields:
                if field in stats:
                    typical = stats[field]['mode']
                    user_val = input_dict[field]
                    if user_val != typical and typical != "N/A":
                        recommendations.append(f"{field} is {user_val}, but the most common for {predicted_crop} is {typical}.")
        recommendation_text = "\n".join(recommendations) if recommendations else "All parameters are within the optimal range for this crop."
        
        if request.headers.get("X-Requested-With") == "XMLHttpRequest":
            return {
                "predicted_crop": predicted_crop,
                "confidence": round(confidence, 1),
                "recommendations": recommendation_text
            }
        
        return render_template("soil.html", {
            "request": request,
            "predicted_crop": predicted_crop,
            "confidence": f"{confidence:.1f}%",
            "recommendations": recommendation_text
        })

    except Exception as e:
        print(f"Prediction Error: {e}")
        if request.headers.get("X-Requested-With") == "XMLHttpRequest":
            return {"error": str(e)}
        return render_template("soil.html", {"request": request, "error": str(e)})

# ---------- Run ----------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8080, log_level="info")