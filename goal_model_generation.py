from fastapi import Request, Form, Depends
from fastapi.templating import Jinja2Templates
from fastapi import APIRouter

from sqlalchemy.orm import Session
from database import SessionLocal, engine
import models
from sqlalchemy.orm import aliased

from transformers.utils import logging
import torch
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
logging.set_verbosity_error()

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



def process_goal_hierarchy(db: Session):
    # Fetch all goals
    all_goals = db.query(models.Goal).all()

    # Fetch all hierarchies
    hierarchies = db.query(models.Hierarchy).all()

    # Process and map the goals with their hierarchies
    hierarchy_data = []

    for goal in all_goals:
        for hierarchy in hierarchies:
            if hierarchy.subgoal_id == goal.id or hierarchy.high_level_goal_id == goal.id:
                hierarchy_data.append({
                    'subgoal_id': hierarchy.subgoal_id,
                    'subgoal_name': db.query(models.Goal).filter(models.Goal.id == hierarchy.subgoal_id).first().goal_name,
                    'subgoal_goal_type': db.query(models.Goal).filter(models.Goal.id == hierarchy.subgoal_id).first().goal_type,
                    'high_level_goal_id': hierarchy.high_level_goal_id,
                    'high_level_goal_name': db.query(models.Goal).filter(
                        models.Goal.id == hierarchy.high_level_goal_id).first().goal_name,
                    'high_level_goal_goal_type': db.query(models.Goal).filter(
                        models.Goal.id == hierarchy.high_level_goal_id).first().goal_type,
                    'refinement': hierarchy.refinement
                })

    # Add single goals with no hierarchy
    for goal in all_goals:
        if not any(goal.id == h['subgoal_id'] or goal.id == h['high_level_goal_id'] for h in hierarchy_data):
            hierarchy_data.append({
                'subgoal_id': goal.id,
                'subgoal_name': goal.goal_name,
                'subgoal_goal_type': goal.goal_type,
                'high_level_goal_id': None,
                'high_level_goal_name': None,
                'high_level_goal_goal_type': None,
                'refinement': None
            })

    return hierarchy_data



@router.get("/goal_model_generation")
async def goal_model_generation(request: Request, db: Session = Depends(get_db)):
    return templates.TemplateResponse('goal_model_generation.html', context={'request': request, 'hierarchy_data': process_goal_hierarchy(db)})



@router.post("/goal_model_generation")
async def deleteGoal(request: Request, subgoal_id: int = Form(...), db: Session = Depends(get_db)):
    # Log
    print("Subgoal ID to delete:", subgoal_id)

    # Query the database to find the subgoal
    subgoal = db.query(models.Goal).filter(models.Goal.id == subgoal_id).first()

    if subgoal:
        # Delete the subgoal and all related hierarchies
        db.query(models.Hierarchy).filter(
            (models.Hierarchy.subgoal_id == subgoal_id) | (models.Hierarchy.high_level_goal_id == subgoal_id)).delete()
        db.delete(subgoal)
        db.commit()

        print("The subgoal ID " + str(subgoal_id) + " is deleted!")

    return templates.TemplateResponse('goal_model_generation.html',
                                      context={'request': request, 'hierarchy_data': process_goal_hierarchy(db)})