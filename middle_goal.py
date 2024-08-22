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
async def middle_goal(request: Request, high_level_goal_id: int, db: Session = Depends(get_db)):
    goal_with_outputs = db.query(models.Goal).filter(models.Goal.id == high_level_goal_id).first()

    if not goal_with_outputs:
        return RedirectResponse("/")

    highlevelgoal = goal_with_outputs.goal_name

    return templates.TemplateResponse('middle_goal.html', context={'request': request, 'hlg_id': high_level_goal_id, 'highlevelgoal': highlevelgoal})


@router.post("/middle_goal")
async def middle_goal(request: Request, goal_name: str = Form(...), goal_type: str = Form(...), hlg_id: int = Form(...),
                      refinement_type: str = Form(...), db: Session = Depends(get_db)):
    # Create the new middle goal
    if hlg_id != -1:
        new_middle_goal = models.Goal(goal_type=goal_type, goal_name=goal_name)
        db.add(new_middle_goal)
        db.commit()
        db.refresh(new_middle_goal)

    # Get all current subgoals linked to the high-level goal
    subgoals = db.query(models.Goal).join(models.Hierarchy, models.Goal.id == models.Hierarchy.subgoal_id).filter(
        models.Hierarchy.high_level_goal_id == hlg_id).all()

    # Separate subgoals by their refinement type (AND/OR/XOR)
    AND_subgoals = []
    OR_subgoals = []
    XOR_subgoals = []

    for subgoal in subgoals:
        hierarchy = db.query(models.Hierarchy).filter_by(high_level_goal_id=hlg_id, subgoal_id=subgoal.id).first()
        if hierarchy.refinement == "AND":
            AND_subgoals.append(subgoal)
        elif hierarchy.refinement == "OR":
            OR_subgoals.append(subgoal)
        elif hierarchy.refinement == "XOR":
            XOR_subgoals.append(subgoal)

    # --> Update the goal hierarchy <--
    # Break the current hierarchy between the high-level goal and subgoals
    if refinement_type == "AND": # If the middle goal is to be added to the AND branch
        new_hierarchy_hlg_and = models.Hierarchy(refinement="AND", high_level_goal_id=hlg_id, subgoal_id=new_middle_goal.id)
        db.add(new_hierarchy_hlg_and)

        for subgoal in AND_subgoals:
            new_hierarchy_sub_and = models.Hierarchy(refinement="AND", high_level_goal_id=new_middle_goal.id, subgoal_id=subgoal.id)
            db.add(new_hierarchy_sub_and)

            # Delete the old hierarchy links
            old_hierarchy_and = db.query(models.Hierarchy).filter(models.Hierarchy.high_level_goal_id == hlg_id,
                                                                  models.Hierarchy.subgoal_id == subgoal.id).first()
            db.delete(old_hierarchy_and)
    elif refinement_type == "OR": # If the middle goal is to be added to the OR branch
        new_hierarchy_hlg_or = models.Hierarchy(refinement="OR", high_level_goal_id=hlg_id, subgoal_id=new_middle_goal.id)
        db.add(new_hierarchy_hlg_or)

        for subgoal in OR_subgoals:
            new_hierarchy_sub_or = models.Hierarchy(refinement="OR", high_level_goal_id=new_middle_goal.id, subgoal_id=subgoal.id)
            db.add(new_hierarchy_sub_or)

            # Delete the old hierarchy links
            old_hierarchy_or = db.query(models.Hierarchy).filter(models.Hierarchy.high_level_goal_id == hlg_id,
                                                                 models.Hierarchy.subgoal_id == subgoal.id).first()
            db.delete(old_hierarchy_or)
    elif refinement_type == "XOR": # If the middle goal is to be added to the XOR branch
        new_hierarchy_hlg_xor = models.Hierarchy(refinement="XOR", high_level_goal_id=hlg_id, subgoal_id=new_middle_goal.id)
        db.add(new_hierarchy_hlg_xor)

        for subgoal in XOR_subgoals:
            new_hierarchy_sub_xor = models.Hierarchy(refinement="XOR", high_level_goal_id=new_middle_goal.id, subgoal_id=subgoal.id)
            db.add(new_hierarchy_sub_xor)

            # Delete the old hierarchy links
            old_hierarchy_xor = db.query(models.Hierarchy).filter(models.Hierarchy.high_level_goal_id == hlg_id,
                                                                 models.Hierarchy.subgoal_id == subgoal.id).first()
            db.delete(old_hierarchy_xor)

    db.commit()

    return RedirectResponse(f"/goal_model_generation", status_code=302)