
# 🏡 Airbnb Paris Predictor

**A Data Science and Machine Learning project** that analyzes Airbnb listings in Paris to **predict nightly prices** and **availability trends** using real-world data from [Inside Airbnb](http://insideairbnb.com/get-the-data/).  
The project builds an end-to-end ML pipeline — from data cleaning and feature engineering to model training and interactive prediction via Streamlit.

---

## 🌍 Project Overview

This project uses publicly available Airbnb data for **Paris, France** to:

- 🧹 Clean and preprocess raw listing data  
- 🧠 Train machine learning models to:
  - Predict listing **price per night**
  - Estimate **availability** over the next year  
- 📊 Deploy interactive dashboards for lay users to explore predictions with simple inputs  
- ⚙️ Provide explainable, user-friendly visualizations of predicted outcomes  

---

## 📦 Tech Stack

| Category | Tools / Libraries |
|-----------|------------------|
| **Languages** | Python (3.10+) |
| **Data Processing** | Pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn |
| **Machine Learning** | Scikit-learn, XGBoost |
| **Model Explainability** | SHAP |
| **App Deployment** | Streamlit |
| **Environment & Utilities** | Joblib, Pathlib, Git, Git LFS |

---

## 📥 Dataset

You can download the official **Paris Airbnb listings dataset** from [Inside Airbnb](http://insideairbnb.com/get-the-data/).

**Direct CSV link (latest Paris data):**  
👉 [http://data.insideairbnb.com/france/ile-de-france/paris/2025-01-04/data/listings.csv.gz](http://data.insideairbnb.com/france/ile-de-france/paris/2025-01-04/data/listings.csv.gz)

Once downloaded:
1. Extract the `.gz` file  
2. Rename it to `listings.csv`  
3. Place it in your project at:

data/raw/listings.csv

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/zainulisalm0001/Airbnb-Paris-Predictor.git
cd Airbnb-Paris-Predictor

2️⃣ Create a Virtual Environment

python -m venv .venv
source .venv/bin/activate   # On macOS/Linux
.venv\Scripts\activate      # On Windows

3️⃣ Install Dependencies

pip install -r requirements.txt

4️⃣ Prepare the Dataset

Download and place listings.csv in:

data/raw/listings.csv

5️⃣ Train Models

Run both model training scripts to generate your model files:

python -m src.train_price_model
python -m src.train_availability_model

This will create:

api/model_price.pkl
api/preprocessor_price.pkl
api/model_availability.pkl
api/preprocessor_availability.pkl


⸻

🖥️ Running the Streamlit App

After training, launch the app locally with:

streamlit run streamlit_app/app.py

Open http://localhost:8501 in your browser.

⸻

🌐 Live Demo (Streamlit Cloud)

You can explore the live deployed version here:
👉 Airbnb Paris Predictor (Streamlit App)
(replace this link with your deployed Streamlit app once live)

⸻

🧠 Features

🏷️ Price Predictor

Estimate Airbnb nightly prices based on:
	•	Number of bedrooms, bathrooms, and beds
	•	Property type and room type
	•	Location (latitude/longitude)
	•	Superhost status and minimum nights

📅 Availability Predictor

Predict whether a property will likely be available based on:
	•	Average minimum nights
	•	Host type
	•	Room type and neighborhood
	•	Booking trends and historical occupancy

⸻

📊 Example Outputs

Price Predictor
	•	Displays predicted nightly price in €
	•	Visual gauge shows low–medium–high price bands

Availability Predictor
	•	Outputs likelihood of the property being bookable (0–100%)
	•	Includes confidence visual indicator

⸻

💡 Future Improvements
	•	Integrate real-time Airbnb API (if available)
	•	Incorporate NLP features (amenity extraction from descriptions)
	•	Add advanced SHAP visual explainability for non-technical users
	•	Extend model to other major cities (e.g., London, New York)

⸻

🧾 License

This project is licensed under the MIT License.

⸻

💬 Acknowledgments

Data source: Inside Airbnb
Built with ❤️ using Python, Streamlit, and XGBoost by Muhammad Zain Ul Islam
GitHub: @zainulisalm0001

⸻
