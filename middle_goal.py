from fastapi import Request, Form, Depends
from fastapi.templating import Jinja2Templates
from fastapi import APIRouter

from typing import List, Optional

from sqlalchemy.orm import Session
from starlette.responses import RedirectResponse

from database import SessionLocal, engine
import models

# Templates (Jinja2)
templates = Jinja2Templates(directory="templates/")

# Router
router = APIRouter()

# Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

models.Base.metadata.create_all(bind=engine)


@router.get("/middle_goal/{high_level_goal_id}")
async def middle_goal(request: Request, high_level_goal_id: int):
    return templates.TemplateResponse('middle_goal.html', context={'request': request, 'hlg_id': high_level_goal_id})


@router.post("/middle_goal")
async def middle_goal(request: Request, goal_name: str = Form(...), goal_type: str = Form(...), hlg_id: int = Form(...), db: Session = Depends(get_db)):
    # Create a new middle goal
    if hlg_id != -1:
        new_middle_goal = models.Goal(goal_type=goal_type, goal_name=goal_name)
        db.add(new_middle_goal)
        db.commit()
        db.refresh(new_middle_goal)

    # --> Update the hierarchy
    # Break the current hierarchy between the high-level goal and subgoals

    # Query to get all subgoals for a specific high-level goal
    subgoals = db.query(models.Goal).join(models.Hierarchy, models.Goal.id == models.Hierarchy.subgoal_id).filter(models.Hierarchy.high_level_goal_id == hlg_id).all()

    # Insert new middle goal (with AND refinements by default)
    # New hierarchy between the existing high-level goal and new middle goal
    new_hierarchy_high = models.Hierarchy(refinement="AND", high_level_goal_id=hlg_id, subgoal_id=new_middle_goal.id)
    db.add(new_hierarchy_high)

    for subgoal in subgoals:
        # New hierarchy between new middle goal and the existing subgoals
        new_hierarchy_sub = models.Hierarchy(refinement="AND", high_level_goal_id=new_middle_goal.id, subgoal_id=subgoal.id)
        db.add(new_hierarchy_sub)

        # Delete old hierarchy
        old_hierarchy = db.query(models.Hierarchy).filter(models.Hierarchy.high_level_goal_id == hlg_id, models.Hierarchy.subgoal_id == subgoal.id).first()
        db.delete(old_hierarchy)
        print("Old hierarchy deleted")

        db.commit()
        print("commited")

    return templates.TemplateResponse('middle_goal.html', context={
        'request': request,
        'hlg_id': hlg_id,
    })