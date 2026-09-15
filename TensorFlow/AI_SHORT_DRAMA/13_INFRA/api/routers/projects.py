"""/projects 路由：项目（一部短剧）的 CRUD。

项目是所有资源的顶层归属：角色、剧集、流水线运行都挂在某个 project_id 下。
整组路由挂了 ``require_api_key`` 依赖，全部需要 X-API-Key。
"""

from __future__ import annotations

from typing import List

from fastapi import APIRouter, Depends, HTTPException, Response
from sqlalchemy.orm import Session

from ...database.models import Project
from ..deps import get_db
from ..pagination import Page, page_params, paginate
from ..schemas import ProjectCreate, ProjectRead
from ..security import require_api_key

router = APIRouter(prefix="/projects", tags=["projects"], dependencies=[Depends(require_api_key)])


@router.post("", response_model=ProjectRead, status_code=201)
def create_project(payload: ProjectCreate, db: Session = Depends(get_db)) -> Project:
    """创建项目；返回 201 和完整记录（refresh 后才有数据库生成的 id 和时间戳）。"""
    project = Project(name=payload.name, description=payload.description)
    db.add(project)
    db.commit()
    db.refresh(project)
    return project


@router.get("", response_model=List[ProjectRead])
def list_projects(response: Response, db: Session = Depends(get_db), page: Page = Depends(page_params)) -> List[Project]:
    """分页列出项目，按创建时间倒序；总数在 X-Total-Count 头里。"""
    return paginate(db.query(Project).order_by(Project.created_at.desc()), page, response)


@router.get("/{project_id}", response_model=ProjectRead)
def get_project(project_id: str, db: Session = Depends(get_db)) -> Project:
    """按 id 取单个项目，不存在返回 404。"""
    project = db.get(Project, project_id)
    if project is None:
        raise HTTPException(status_code=404, detail="Project not found")
    return project


@router.delete("/{project_id}", status_code=204)
def delete_project(project_id: str, db: Session = Depends(get_db)) -> None:
    """删除项目（关联记录按 ORM 的级联规则处理），成功返回 204 无正文。"""
    project = db.get(Project, project_id)
    if project is None:
        raise HTTPException(status_code=404, detail="Project not found")
    db.delete(project)
    db.commit()
