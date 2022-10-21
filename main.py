from fastapi import FastAPI, UploadFile
from predictor import predict
import os

app = FastAPI()
voice_url = '/identify-voice'

@app.get('/')
def index():
    return {"message": f"Hi, this is an API that identifies voices. You might want to send a POST request to {voice_url} to see us in action."}

@app.post('/test-file')
def test_file(file: UploadFile):
    return {"success": True, "message": f"Received file {file.filename} of content type {file.content_type}.", "prediction": ""}

@app.post(voice_url)
def identify_voice(audio: UploadFile):
    if len(audio.filename) == 0:
        return { "success": False, "message": "Please upload a valid audio file.", "prediction": "" }

    # save the audio file
    with open(audio.filename, 'wb') as f:
        f.write(audio.file.read())

    # get prediction
    prediction = predict(audio.filename)    

    # delete the temporary file
    os.remove(audio.filename)

    return { "success": True, "message": "Prediction successful", "prediction": prediction }