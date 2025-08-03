import os
from fastapi import Request, UploadFile, File, APIRouter
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from graph_extender import graph_extender
import requests

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

        # Upload to GraphDB using requests
        with open(output_path, "rb") as data:
            response = requests.post(
                "http://localhost:7200/repositories/Flood_Management_KG/statements",
                data=data,
                headers={"Content-Type": "application/rdf+xml"}
            )

        if response.status_code in [200, 204]:
            message = f"Graph uploaded and extended successfully! Saved to: {output_path}"
        else:
            message = f"Failed to upload to GraphDB: {response.status_code} - {response.text}"

    except Exception as e:
        message = f"Failed to process the graph: {str(e)}"
    finally:
        if os.path.exists(uploaded_path):
            os.remove(uploaded_path)

    return templates.TemplateResponse("upload_kg.html", {"request": request, "message": message})
