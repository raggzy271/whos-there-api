from fastapi import FastAPI, UploadFile
from predictor import predict
import os

app = FastAPI()

@app.get('/')
def index():
    return {"message": f"Hi, this is an API that identifies voices. You might want to send a POST request to /identify-voice to see us in action."}

@app.post('/identify-voice')
async def identify_voice(audio: UploadFile):
    path_3gp = audio.filename
    if len(path_3gp) == 0:
        return { "success": False, "message": "Please upload a valid audio file.", "prediction": "" }

    # save the audio file
    with open(path_3gp, 'wb') as f:
        f.write(audio.file.read())

    # convert 3gp to wav file
    path_wav = path_3gp.replace('.3gp', '.wav')
    os.system(f"ffmpeg -i {path_3gp} {path_wav}")

    # get prediction
    prediction = predict(path_wav)    

    # delete the files
    os.remove(path_3gp)
    os.remove(path_wav)

    return { "success": True, "message": "Prediction successful", "prediction": prediction }