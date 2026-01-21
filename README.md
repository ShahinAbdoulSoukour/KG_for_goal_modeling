[![python](https://img.shields.io/badge/python-3.9%20|%203.10%20|%203.11-blue.svg?style=flat&logo=python&logoColor=white)](https://www.python.org)

## Installing the dependencies

Inside a dedicated Python environment:

```shell
pip install -r requirements.txt
```

## Installing GraphDB

1.  Download GraphDB distribution: 
```shell
curl -L -o graphdb.zip https://download.ontotext.com/owlim/dbba7356-57e2-11f0-aa4c-42843b1b6b38/graphdb-11.0.2-dist.zip
```
2.	Install brew:
```shell
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```
3.  Add Homebrew to your PATH:
```shell
eval "$(/home/linuxbrew/.linuxbrew/bin/brew shellenv)"
```
4.  Unzip the GraphDB package:
```shell
unzip graphdb.zip
```
5.  Install jdk
```shell
brew install openjdk
```
6.  Start GraphDB:
```shell
./bin/graphdb
```
7.  Add a license and create a new repository through the GraphDB interface (via Settings).
Use a name matching your project, for example:
```shell
Flood_Management_KG
```
Ensure that this repository name matches the query and update endpoints defined in `contextualization.py`:
```shell
QUERY_ENDPOINT = "http://localhost:7200/repositories/Flood_Management_KG"
UPDATE_ENDPOINT = "http://localhost:7200/repositories/Flood_Management_KG/statements"
```
And in `upload_kg.py`:
```shell
# Upload to GraphDB using requests
with open(output_path, "rb") as data:
    response = requests.post(
        "http://localhost:7200/repositories/Flood_Management_KG/statements",
        data=data,
        headers={"Content-Type": "application/rdf+xml"}
    )
```

## Run the tool

```shell
uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

The tool is then accessible by opening a webpage at the URL [127.0.0.1:8000](http://127.0.0.1:8000) or [localhost:8000](http://localhost:8000)

## Using HuggingFace Inference Endpoints

If you use HuggingFace Inference Endpoints, you can perform the NLI and sentiment analysis tasks on remote servers by creating a `.env` file at the root of this project and adding the following environment variables:

- `HF_TOKEN`: Your HuggingFace Inference Endpoints access token
- `API_URL_NLI`: The URL to your endpoint containing a model dedicated to NLI (we use `ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli`)
- `API_URL_SENT`: The URL to your endpoint containing a model dedicated to sentiment analysis (we use `cardiffnlp/twitter-roberta-base-sentiment-latest`)

## Accelerating Goal Refinement

To speed up the entire goal refinement process, you can leverage a high-performance GPU by using, for example, the [Runpod](https://www.runpod.io/) platform.
After setting up a pod with all required dependencies and GraphDB installed, you can securely access the environment from your local machine by creating an SSH tunnel and opening it in your web browser.

```shell
ssh -L 8000:127.0.0.1:8000 <your SSH command accessible from the Runpod platform>
```
Now you can open your web browser and access our tool via this link: [127.0.0.1:8000](http://127.0.0.1:8000) or [localhost:8000](http://localhost:8000).

## Citation

```bibtex
@phdthesis{abdoulsoukour:tel-05448495,
  TITLE = {{Leveraging domain knowledge in software system goal models}},
  AUTHOR = {Abdoul Soukour, Shahin},
  URL = {https://theses.hal.science/tel-05448495},
  NUMBER = {2025SORUS359},
  SCHOOL = {{Sorbonne Universit{\'e}}},
  YEAR = {2025},
  MONTH = Sep,
  KEYWORDS = {Natural Language Processing ; Knowledge Graph ; Requirement engineering ; Software engineering ; Traitement automatique du langage naturel ; Graphe de connaissance ; Ing{\'e}nierie des exigences ; G{\'e}nie logiciel},
  TYPE = {Theses},
  PDF = {https://theses.hal.science/tel-05448495v1/file/142066_ABDOUL_SOUKOUR_2025_archivage.pdf},
  HAL_ID = {tel-05448495},
  HAL_VERSION = {v1},
}
```

