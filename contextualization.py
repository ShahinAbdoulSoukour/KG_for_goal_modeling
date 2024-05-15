from fastapi import FastAPI, Request, Form, status, HTTPException, Depends
from fastapi.templating import Jinja2Templates
from fastapi.responses import RedirectResponse, HTMLResponse
from fastapi import APIRouter

from typing import Optional, List

from sqlalchemy.orm import Session
from database import SessionLocal, engine
import models

import pandas as pd
import rdflib
from transformers import AutoTokenizer, AutoModelForSequenceClassification, T5ForConditionalGeneration, T5Tokenizer, pipeline
from transformers.utils import logging
from sentence_transformers import SentenceTransformer
import torch
import os
from anchor_points_extractor import anchor_points_extractor
from utils.sparql_queries import find_all_triples_q
from utils import triple_sentiment_analysis, test_entailment, get_neighbors
from graph_explorator import graph_explorator
from g2t_generator import g2t_generator

os.environ["TOKENIZERS_PARALLELISM"] = "false"

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
logging.set_verbosity_error()


# --- Import models ---
model_sts = SentenceTransformer('all-mpnet-base-v2')

model_nli_name = "ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli"
tokenizer_nli = AutoTokenizer.from_pretrained(model_nli_name)
model_nli = AutoModelForSequenceClassification.from_pretrained(model_nli_name).to(device)

model_g2t = T5ForConditionalGeneration.from_pretrained("Inria-CEDAR/WebNLG20T5B").to(device)
tokenizer_g2t = T5Tokenizer.from_pretrained("t5-base", model_max_length=512)

sentiment_model_path = "cardiffnlp/twitter-roberta-base-sentiment-latest"
sentiment_task = pipeline("sentiment-analysis", model=sentiment_model_path, tokenizer=sentiment_model_path, device=device)

# --- Import the Knowledge Graph (KG) ---
# TODO: upload a knowledge graph (RDF file)
domain_graph = rdflib.Graph()
domain_graph.parse("./flooding_graph.rdf")

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

@router.get("/")
async def contextualization(request: Request):
    return templates.TemplateResponse('contextualization.html', context={'request': request})


@router.post("/")
async def contextualization(request: Request, highlevelgoal: str = Form(...), filtered_out_triples: List[str] = Form([]), db: Session = Depends(get_db)): ## TODO: list of lists
    # clear db
    db.query(models.Anchor_Points).delete()
    db.query(models.Entailment_Results).delete()
    db.commit()

    # Process the received data, for example, print it
    print("Checked triples:", filtered_out_triples)

    modified_filtered_triples = []
    for t in filtered_out_triples:
        modified_filtered_triples.append(t.replace("', '", ' ').replace("['", "").replace("']", ""))

    print("\nmodified triples :")
    print(modified_filtered_triples)

    # --- Extract all triples in the KG ---
    query_results = domain_graph.query(find_all_triples_q)
    triples = [list(map(str, [row["subject"], row["predicate"], row["object"]])) for row in query_results.bindings]

    data = []
    for t in triples:
        subject = t[0]
        predicate = t[1]
        object = t[2]

        # simple triple
        triple = " ".join(t)
        # triples serialized
        triple_with_separator = [t]
        list_goal_triples = [(triple, highlevelgoal, triple_with_separator, subject, predicate, object)]

        for element in list_goal_triples:
            row = {'TRIPLE': element[0], 'GOAL': element[1], 'TRIPLE_SERIALIZED': element[2], 'SUBJECT': element[3],
                   'PREDICATE': element[4], 'OBJECT': element[5]}
            data.append(row)

    goal_triples_df = pd.DataFrame(data)
    print(goal_triples_df)

    # --- Extract anchor points ---
    anchor_points_df = anchor_points_extractor(goal_triples_df, model_sts, modified_filtered_triples).copy()
    print("\nANCHOR TRIPLES:")
    print(anchor_points_df)
    print("\n")

    # Add anchor points in the database (table: anchor_points)
    for index, row in anchor_points_df.iterrows():
        new_anchor_points = models.Anchor_Points(triple = row["TRIPLE"],
                                                     goal = row["GOAL"],
                                                     #triple_serialized = , ###
                                                     subject = row["SUBJECT"],
                                                     predicate = row["PREDICATE"],
                                                     object = row["OBJECT"])
        new_anchor_points.set_triple_serialized(row["TRIPLE_SERIALIZED"])
        db.add(new_anchor_points)
    db.commit()

    # --- Transform negative triples ---
    anchor_points_df["SENTIMENT"] = anchor_points_df["TRIPLE_SERIALIZED"].apply(
        lambda triple: triple_sentiment_analysis(triple[0], sentiment_task, neutral_predicates=["is a type of"])[0])
    anchor_points_df.rename(columns={'TRIPLE': 'PREMISE', 'GOAL': 'HYPOTHESIS'}, inplace=True)

    transformed_triples_premise = []

    for triple, sentiment in zip(anchor_points_df["PREMISE"], anchor_points_df["SENTIMENT"]):
        if sentiment == "negative":
            ### Transformation
            transformed_triples_premise.append("Prevent that " + triple)
        else:
            transformed_triples_premise.append(triple)

    transformed_anchor_points = pd.DataFrame(transformed_triples_premise, columns=["PREMISE"])
    transformed_anchor_points["HYPOTHESIS"] = anchor_points_df["HYPOTHESIS"].values

    transformed_anchor_points["PREMISE_SERIALIZED"] = anchor_points_df["TRIPLE_SERIALIZED"].values
    transformed_anchor_points["SUBJECT"] = anchor_points_df["SUBJECT"].values
    transformed_anchor_points["PREDICATE"] = anchor_points_df["PREDICATE"].values
    transformed_anchor_points["OBJECT"] = anchor_points_df["OBJECT"].values

    # --- Test the entailment between the high-level goal (as hypothesis) and triples (as premise) ---
    entailment_result = test_entailment(transformed_anchor_points, tokenizer_nli, model_nli)
    print("ENTAILMENT RESULTS:")
    print(entailment_result)
    print("\n")

    # Add entailment results in the database (table: entailment_results)
    #for index, row in entailment_result.iterrows():
    #    new_results = models.Entailment_Results(premise=row["PREMISE"],
    #                                             hypothesis=row["HYPOTHESIS"],
    #                                             #premise_serialized=row["PREMISE_SERIALIZED"],
    #                                             subject=row["SUBJECT"],
    #                                             predicate=row["PREDICATE"],
    #                                             object=row["OBJECT"],
    #                                             entailment=row["ENTAILMENT"],
    #                                             neutral=row["NEUTRAL"],
    #                                             contradiction=row["CONTRADICTION"],
    #                                             nli_label=row["NLI_LABEL"])
    #    new_results.set_triple_serialized(row["PREMISE_SERIALIZED"])
    #    db.add(new_results)
    #db.commit()

    # --- Explore graph to improve contextualization ---
    entailed_triples_df = graph_explorator(entailment_result, highlevelgoal, domain_graph, tokenizer_nli, model_nli)

    # --- Generate text ---
    all_triples_entailed = [triple for triples in entailed_triples_df["SUBGOALS_SERIALIZED"].tolist() for triple in
                            triples]

    if not entailed_triples_df.empty:
        all_triples_entailed.append(triples[0] for triples in entailed_triples_df["SUBGOALS_SERIALIZED"].tolist())

    unique_triples_entailed = []
    for triple in all_triples_entailed:
        if (triple not in unique_triples_entailed) and (type(triple) is list):
            unique_triples_entailed.append(triple)

    print("UNIQUE TRIPLES:")
    print(unique_triples_entailed)
    print("\n")

    triples_already_processed = []
    predictions = []

    for idx, row in entailed_triples_df.iterrows():
        triples_to_process = []
        for triple in row["SUBGOALS_SERIALIZED"]:
            if (triple not in triples_already_processed) and (type(triple) is list):
                triples_to_process.append(triple)

        if triples_to_process:
            # --> Generate text
            prediction = g2t_generator(triples_to_process, model=model_g2t, tokenizer=tokenizer_g2t)[0]
            text_version = row["SUBGOALS"].split(". ")[-1]
            # Add the type of goal as a prefix at the beginning of the text
            if "Prevent that" in text_version:
                prediction = "[AVOID] " + prediction
            else:
                prediction = "[ACHIEVE] " + prediction

            print("PREDICTIONS:")
            print(prediction)
            print("\n")

            predictions.append(prediction)

        triples_already_processed.extend(triples_to_process)

    if unique_triples_entailed:
        return templates.TemplateResponse('contextualization.html', context={'request': request,
                                                                            'highlevelgoal': highlevelgoal,
                                                                            'entailed_triples': entailed_triples_df,
                                                                            'unique_triples_entailed': enumerate(unique_triples_entailed),
                                                                            'predictions': predictions,
                                                                            })
    else:
        unique_triples_entailed = []
        return templates.TemplateResponse('contextualization.html', context={'request': request,
                                                                             'highlevelgoal': highlevelgoal,
                                                                             'unique_triples_entailed': unique_triples_entailed})
