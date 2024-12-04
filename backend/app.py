# app.py

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI


client = OpenAI(api_key="[OpenAI Key Here]")

# Load the trained model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("./model")  # Load from the directory where the model is saved
model = AutoModelForSequenceClassification.from_pretrained("./model")

# Initialize the FastAPI app
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5194"],  # Allow all origins, or specify specific origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods, or specify methods like ["GET", "POST"]
    allow_headers=["*"],  # Allow all headers, or specify headers
)

# Define a Pydantic model to validate input
class TextInput(BaseModel):
    text: str

def classify_input_text(input_text: str):
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    prediction = outputs.logits.argmax(dim=-1).item()  # Get the predicted class

    return "spam" if prediction == 1 else "ham"

@app.post("/classify")
async def classify_text(input: TextInput):
    try:
       classification = classify_input_text(input.text)

       if classification == "ham":
        completion = client.chat.completions.create(
                model="gpt-3.5-turbo", 
                messages=[{"role": "user", "content": input.text}],
                max_tokens=1000
            )

        return {
               "text": input.text,
               "predicted_class": classification,
               "openai_response": completion.choices[0].message.content.strip()
               }
       else:
        return {
                    "text": input.text,
                    "predicted_class": classification,
                    "openai_response": "This message is classified as spam."
                }

    except Exception as e:
        # Handle any errors that occur during inference
        print(f"Error {e}")
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")