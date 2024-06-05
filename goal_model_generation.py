from fastapi import Request, Depends
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


@router.get("/goal_model_generation")
async def goal_model_generation(request: Request, db: Session = Depends(get_db)):
    high_level_goal = aliased(models.Goal)
    subgoal_goal = aliased(models.Goal)
    outputs = aliased(models.Outputs)
    triple_filtered = aliased(models.Triple_Filtered)

    goal_hierarchy = db.query(
        models.Hierarchy,
        high_level_goal.id.label('high_level_goal_id'),
        high_level_goal.goal_name.label('high_level_goal_name'),
        subgoal_goal.id.label('subgoal_id'),
        subgoal_goal.goal_name.label('subgoal_name'),
        outputs.entailed_triple.label('entailed_triple'),
        triple_filtered.triple_filtered_from_hlg.label('filtered_triple')
    ).join(
        high_level_goal, models.Hierarchy.high_level_goal_id == high_level_goal.id
    ).join(
        subgoal_goal, models.Hierarchy.subgoal_id == subgoal_goal.id
    ).join(
        outputs, high_level_goal.id == outputs.goal_id
    ).join(
        triple_filtered, subgoal_goal.id == triple_filtered.subgoal_id
    ).all()

    print(goal_hierarchy)

    # Process the goal hierarchy to remove redundant filtered triples
    hierarchy_data = []
    combined_data = {}

    for h in goal_hierarchy:
        key = (h.high_level_goal_name, h.subgoal_name)
        if key not in combined_data:
            print(key)
            combined_data[key] = {
                'high_level_goal_id': h.high_level_goal_id,
                'high_level_goal_name': h.high_level_goal_name,
                'subgoal_id': h.subgoal_id,
                'subgoal_name': h.subgoal_name,
                'filtered_triples': set(),
                'entailed_triples': set()
            }
        if h.filtered_triple:
            combined_data[key]['filtered_triples'].add(h.filtered_triple)
        if h.entailed_triple:
            combined_data[key]['entailed_triples'].add(h.entailed_triple)

    for key, value in combined_data.items():
        hierarchy_data.append({
            'high_level_goal_id': value['high_level_goal_id'],
            'high_level_goal_name': value['high_level_goal_name'],
            'subgoal_id': value['subgoal_id'],
            'subgoal_name': value['subgoal_name'],
            'filtered_triples': list(value['filtered_triples']),
            'entailed_triples': list(value['entailed_triples'])
        })

    return templates.TemplateResponse('goal_model_generation.html', context={'request': request, 'hierarchy_data': hierarchy_data})
