import os
from fastapi import Request, UploadFile, File, APIRouter
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from graph_extender import graph_extender

# Router and templates
router = APIRouter()
templates = Jinja2Templates(directory="templates/")

# Folder to save extended graphs
EXTENDED_FOLDER = "extended_graphs"
os.makedirs(EXTENDED_FOLDER, exist_ok=True)

@router.get("/upload_kg", response_class=HTMLResponse)
async def upload_kg_form(request: Request):
    return templates.TemplateResponse("upload_kg.html", {"request": request})

@router.post("/upload_kg")
async def upload_kg(request: Request, file: UploadFile = File(...)):
    # Save uploaded file temporarily
    uploaded_path = f"temp_{file.filename}"
    with open(uploaded_path, "wb") as f:
        f.write(await file.read())

    # Extend the graph
    try:
        extended_graph = graph_extender(uploaded_path)

        # Save extended graph
        output_path = os.path.join(EXTENDED_FOLDER, f"extended_{file.filename}")
        extended_graph.serialize(destination=output_path, format="xml")

        message = f"Graph uploaded and extended successfully! Saved to: {output_path}"
    except Exception as e:
        message = f"Failed to process the graph: {str(e)}"
    finally:
        os.remove(uploaded_path)

    return templates.TemplateResponse("upload_kg.html", {"request": request, "message": message})
