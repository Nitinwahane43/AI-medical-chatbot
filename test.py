import google.generativeai as genai
from dotenv import load_dotenv
import os

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

model = genai.GenerativeModel("gemini-1.5-flash")

response = model.generate_content("What is a healthy blood pressure?")
print("✅ Gemini says:\n", response.text)
