from fastapi import FastAPI, UploadFile, File
from openai import OpenAI
import base64

app = FastAPI()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image_base64 = base64.b64encode(image_bytes).decode()

    response = client.responses.create(
        model="gpt-4.1-mini",
        input=[{
            "role":"user",
            "content":[
                {"type":"input_text","text":"Describe what this device sees in one short sentence."},
                {"type":"input_image","image_base64":image_base64}
            ]
        }]
    )

    return {"result": response.output_text}