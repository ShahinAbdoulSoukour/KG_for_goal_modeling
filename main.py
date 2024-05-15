from fastapi import FastAPI

import contextualization
#import goal_relaxation
#import exploration

app = FastAPI(title="KG for Goal Model Generation")

app.include_router(contextualization.router)
#app.include_router(goal_relaxation.router)
#app.include_router(exploration.router)