from enum import Enum

from fastapi import FastAPI, Path
import uvicorn
from pyparsing import Optional
from pydantic import BaseModel,Field

app=FastAPI()



class Gender(str,Enum):
    male="男"
    female="女"

class UserModel (BaseModel) :
    username: str
    description: Optional[str] = None
    gender: Gender
class User (BaseModel):
    username: str = Field( ..., min_length=3)
    description: Optional[str] = Field ( None, max_Length=10)
    address: UserModel
@app.get('/user/{user_id}')
async def get_user(user_id:int):
    return {'user_id':user_id}

# 使用枚举方式进行选项列表
@app.get('/user/{gender}')
async def get_gender(gender:Gender):
    return {'gender':gender}

@app.get ('/users')
async def get_user (page_index: int, page_size: int):
    return {'page info': f'index: {page_index} size:  {page_size}'}

@app.get('/users/{user_id)/friends')
async def get_user_friendspage_index( page_index:int, user_id: int, page_size: Optional[int] = 10):#默认值Optional[int] = 10
    # 默认值Optional[str] = None。 不能为空
    return {f'user friends': f'user id: {user_id}, index: {page_index}, size: {page_size}'}

@app.put('/user}')
async def get_gender(userModel:UserModel):
    userDict=userModel.model_dump()
    return {'userDict':userDict}

@app.post('/user}')
async def get_gender(userModel:UserModel):
    userDict=userModel.model_dump()
    return {'userDict':userDict}


#路径参数都是必须项
@app.get ('/users/ (user_id) ')
#ge大于。 le小于。 ...取决于Path的必选
async def get_user (user_id: int = Path(...,title="The user id",ge=1,le=10000)) :
    return {'user': f'This is the user for (user_id) '}
if __name__=='__main__':
    uvicorn.run('main:app',reload=True)