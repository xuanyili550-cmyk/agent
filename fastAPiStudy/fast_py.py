from  fastapi import FastAPI

app=FastAPI()
@app.get("/hello")
async def index():
    return {'messger':'hello word '}

@app.get("/")
async def index():
    return {'messger':'hello word 1 '}

@app.get('/user/{user_id}')
async def get_user(user_id:int):
    return {'user_id':user_id}