from fastapi import FastAPI

import contextualization
import goal_model_generation

app = FastAPI(title="Knowledge Graph for Goal Model Generation")

app.include_router(contextualization.router)
app.include_router(goal_model_generation.router)

# TODO: goal relaxation
