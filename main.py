from fastapi import FastAPI, UploadFile, File
from openai import OpenAI
from dotenv import load_dotenv
import os
import base64

load_dotenv()

app = FastAPI()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    # read image bytes
    image_bytes = await file.read()
    # encode as Base64
    base64_image = base64.b64encode(image_bytes).decode("utf-8")

    response = client.responses.create(
        model="gpt-4.1-mini",
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "Extract the text from the provided image, assuming it contains a question or problem statement. Then, provide a clear, concise answer to that question or problem. If the text is unclear or incomplete, note any assumptions you make. Keep your response factual and step-by-step if it involves reasoning or calculations."},
                    {"type": "input_image", "image_url": f"data:image/jpeg;base64,{base64_image}"}
                ]
            }
        ]
    )

    return {"result": response.output_text}
