from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class UserRequest(BaseModel):
    name: str
    
@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/test")
async def say_hi(request: UserRequest):
    return {"message": f"Hi {request.name}, I am Python server!"}