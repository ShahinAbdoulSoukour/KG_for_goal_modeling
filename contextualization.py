from fastapi import Request, Form, Depends
from fastapi.templating import Jinja2Templates
from fastapi import APIRouter

from typing import List

from sqlalchemy.orm import Session
from starlette.responses import RedirectResponse

from database import SessionLocal, engine
import models

import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, T5ForConditionalGeneration, T5Tokenizer, pipeline
from transformers.utils import logging
from sentence_transformers import SentenceTransformer
import torch
import os
from anchor_points_extractor import anchor_points_extractor
from utils.sparql_queries import find_all_triples_q
from utils import triple_sentiment_analysis, test_entailment, triple_sentiment_analysis_api
from graph_explorator import graph_explorator
from g2t_generator import g2t_generator
from graph_extender import graph_extender

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


# --- Import the Knowledge Graph (KG) ---
# TODO: upload a knowledge graph (RDF file)
domain_graph = graph_extender("./flooding_graph_V2.rdf")
# domain_graph = rdflib.Graph()
# domain_graph.parse("./flooding_graph.rdf")


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


def get_subgoal_branch_from_triple_filtered(subgoal_id, db: Session):
    # Initialize the path and the stack
    path = [] # To store the results
    stack = [(subgoal_id, [])] # Initialized with the subgoal_id and an empty path

    # For each current_goal_id in the stack, the corresponding Goal is retrieved
    while stack:
        current_goal_id, current_path = stack.pop()
        current_goal = db.query(models.Goal).filter_by(id=current_goal_id).first()

        if current_goal is None:
            continue

        # Retrieve the filtered triples that created the current goal (--> the subgoal)
        # from the table: Triple_Filtered
        filtered_triple_list = []
        filtered_triples = db.query(models.Triple_Filtered).filter_by(subgoal_id=current_goal_id).all()

        if filtered_triples:
            for triple in filtered_triples:
                filtered_triple_list.append({
                    'high_level_goal_id': triple.high_level_goal_id,
                    'entailed_triples': triple.get_entailed_triples()
                })

        # The new path is constructed by prepending the current goal and its filtered triples to the current_path
        new_path = [(current_goal, filtered_triple_list)] + current_path

        # Updating the path
        # The path is updated with the new path
        path = new_path

        # Check for high-level goals
        # High-level goals for the current goal are retrieved (from the table: Hierarchy), and each high-level goal is added to the stack
        high_level_goals = db.query(models.Hierarchy).filter_by(subgoal_id=current_goal_id).all()
        for hlg in high_level_goals:
            stack.append((hlg.high_level_goal_id, new_path))

    # path --> it contains the subgoal, the filtered triples used to create it, and its high-level goals in the correct order
    return path


@router.get("/")
async def contextualization(request: Request, db: Session = Depends(get_db)):
    all_goal = db.query(models.Goal).all()
    return templates.TemplateResponse('contextualization.html', context={'request': request, 'all_goal': all_goal})


@router.get("/contextualization/{hlg_id}")
async def contextualization(request: Request, hlg_id: int, db: Session = Depends(get_db)):
    all_goal = db.query(models.Goal).all()

    goal_with_outputs = db.query(models.Goal).filter(models.Goal.id == hlg_id).first()

    if not goal_with_outputs:
        return RedirectResponse("/")

    highlevelgoal = goal_with_outputs.goal_name

    data = []

    for output in goal_with_outputs.outputs:
        data.append({
            'id': output.id,
            'goal_id': output.goal_id,
            'generated_text': output.generated_text,
            'entailed_triple': output.get_entailed_triples()
        })

    # Create a DataFrame for storing all outputs
    outputs_df = pd.DataFrame(data)

    return templates.TemplateResponse('contextualization.html', context={'request': request,
                                                                         'highlevelgoal': highlevelgoal,
                                                                         'outputs': outputs_df,
                                                                         'goal_with_outputs': goal_with_outputs,
                                                                         'all_goal': all_goal})


@router.post("/")
async def contextualization(request: Request, goal_type: str = Form(...), highlevelgoal: str = Form(...), filtered_out_triples_with_goal_id: List[str] = Form([]), db: Session = Depends(get_db)):
    all_goal = db.query(models.Goal).all()

    goal_with_outputs = db.query(models.Goal).filter(models.Goal.goal_name == highlevelgoal).first()

    # If the high-level goal do not exist in the database
    if not goal_with_outputs:
        modified_filtered_triples = []

        # If there are some filtered triples, get the high-level goal id and the triples
        if filtered_out_triples_with_goal_id:
            triples_with_ids = []
            for item in filtered_out_triples_with_goal_id:
                goal_id, triple = item.split('_', 1)
                triples_with_ids.append({
                    'goal_id': goal_id, # High-level goal ID
                    'filtered_triple': triple # Filtered triples (selected by the designer)
                })
            triples_with_ids_df = pd.DataFrame(triples_with_ids) # Add the elements in a dataframe

            print("\nTRIPLES WITH IDS:")
            print(triples_with_ids_df.to_string())

            # If the dataframe is not empty
            if not triples_with_ids_df.empty:
                for row in triples_with_ids_df.itertuples():
                    print(row.filtered_triple)
                    modified_filtered_triples.append({
                        'high_level_goal_id': row.goal_id,  # High-level goal ID
                        'triple_filtered_from_hlg': row.filtered_triple.replace("', '", ' ').replace("['", "").replace("']", "")
                    })

                modified_filtered_triples_df = pd.DataFrame(modified_filtered_triples)

                subgoal_id = modified_filtered_triples_df.loc[0, 'high_level_goal_id'] # Replace with the actual subgoal id
                subgoal_branch = get_subgoal_branch_from_triple_filtered(subgoal_id, db)

                print("\n")

                ###
                # Iterate through the subgoal_branch and append new rows to modified_filtered_triples
                for level, (goal, ft) in enumerate(subgoal_branch):
                    # Ensure ft is a list
                    if isinstance(ft, str):
                        ft = [ft]

                    if ft:
                        for triple_dict in ft:
                            triple = triple_dict['entailed_triples']
                            high_level_goal_id = triple_dict['high_level_goal_id']

                            # Check for redundancy before adding
                            if not any(d['triple_filtered_from_hlg'] == triple and d[
                                'high_level_goal_id'] == high_level_goal_id for d in modified_filtered_triples):
                                modified_filtered_triples.append({
                                    'high_level_goal_id': high_level_goal_id,
                                    'triple_filtered_from_hlg': triple
                                })

                # Convert the updated list to a DataFrame
                modified_filtered_triples_df = pd.DataFrame(modified_filtered_triples)
                print("\nUpdated MODIFIED TRIPLE (STRING TO LIST):")
                print(modified_filtered_triples_df.to_string())

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

        ft = [item['triple_filtered_from_hlg'] for item in modified_filtered_triples]
        print("\nFILTERED TRIPLE(S):")
        print(ft)

        # --- Extract anchor points ---
        anchor_points_df = anchor_points_extractor(goal_triples_df, model_sts, ft).copy()
        print("\nANCHOR TRIPLES:")
        print(anchor_points_df)

        # --- Transform negative triples ---
        anchor_points_df["SENTIMENT"] = anchor_points_df["TRIPLE_SERIALIZED"].apply(
            lambda triple: triple_sentiment_analysis_api(triple[0], neutral_predicates=["is a type of"])[0])
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
        print("\nENTAILMENT RESULTS:")
        print(entailment_result)

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

        print("\nUNIQUE TRIPLES:")
        print(unique_triples_entailed)

        triples_already_processed = []
        processed_data = []
        triples_to_process_grouped = []
        group_is_avoid_list = []

        for row in entailed_triples_df.itertuples():
            triples_to_process = []

            text_version = row.SUBGOALS.split(". ")[-1]
            group_is_avoid = "Prevent that" in text_version
            group_is_avoid_list.append(group_is_avoid)

            for triple in row.SUBGOALS_SERIALIZED:
                if (triple not in triples_already_processed) and (type(triple) is list):
                    triples_to_process.append(triple)
            if triples_to_process:
                triples_to_process_grouped.append(triples_to_process)

            triples_already_processed.extend(triples_to_process)

        if triples_to_process_grouped:
            predictions = g2t_generator(triples_to_process_grouped, model=model_g2t, tokenizer=tokenizer_g2t)

            for is_avoid, prediction, triples_to_process in zip(group_is_avoid_list, predictions, triples_to_process_grouped):
                if is_avoid:
                    prediction = "[AVOID] " + prediction
                else:
                    prediction = "[ACHIEVE] " + prediction

                processed_data.append({
                    "ENTAILED_TRIPLE": triples_to_process,
                    "GENERATED_TEXT": prediction
                })

        # Create DataFrame from the list of dictionaries
        processed_data_df = pd.DataFrame(processed_data)

        if unique_triples_entailed:
            if not modified_filtered_triples:
                # Add the goal (as high-level goal) in the database (table: goal)
                new_goal = models.Goal(goal_type=goal_type, goal_name=highlevelgoal)
                db.add(new_goal)
                db.commit()

                # Add the entailed triples and the generated text in the database (table: outputs)
                for row in processed_data_df.itertuples():
                    new_results = models.Outputs(generated_text=row.GENERATED_TEXT,
                                                 goal_id=new_goal.id)
                    new_results.set_entailed_triple(row.ENTAILED_TRIPLE)
                    db.add(new_results)
                db.commit()

                print("\nHigh-level goal added in the database!")
            else:
                # Add the goal (as subgoal) in the database (table: goal)
                new_goal = models.Goal(goal_type=goal_type, goal_name=highlevelgoal)
                db.add(new_goal)
                db.commit()

                # Add the entailed triples and the generated text in the database (table: outputs)
                for row in processed_data_df.itertuples():
                    new_results = models.Outputs(generated_text=row.GENERATED_TEXT, goal_id=new_goal.id)
                    new_results.set_entailed_triple(row.ENTAILED_TRIPLE)
                    db.add(new_results)
                db.commit()

                for row in modified_filtered_triples_df.itertuples():
                    # Add the filtered triples (selected by the designer for creating subgoals) to the database
                    # (table: filtered_triple)
                    filtered_triple = models.Triple_Filtered(subgoal_id=new_goal.id, high_level_goal_id=row.high_level_goal_id)
                    filtered_triple.set_entailed_triple(row.triple_filtered_from_hlg)
                    db.add(filtered_triple)

                db.commit()
                print("\nSubgoal added in the database!")

                # Add the high-level goal and the subgoal in the database (table: hierarchy)
                db_hierarchy = models.Hierarchy(high_level_goal_id=goal_id, subgoal_id=new_goal.id)
                db.add(db_hierarchy)
                db.commit()
                print("\nUpdate the hierarchy!")

            # Extract the entailed triples and the generated texts (to print)
            outputs = db.query(models.Outputs).filter(models.Outputs.goal_id == new_goal.id).all()

            # Extract data into a list of dictionaries
            data = []
            for output in outputs:
                data.append({
                    'id': output.id,
                    'goal_id': output.goal_id,
                    'generated_text': output.generated_text,
                    'entailed_triple': output.get_entailed_triples()
                })

            # Create a DataFrame for storing all outputs
            outputs_df = pd.DataFrame(data)

            return templates.TemplateResponse('contextualization.html', context={
                'request': request,
                'highlevelgoal': highlevelgoal,
                'unique_triples_entailed': enumerate(unique_triples_entailed),
                'outputs': outputs_df,
                'goal_with_outputs': goal_with_outputs,
                'all_goal': all_goal,  # for the input (for autocompletion)
            })
        else:
            message = "No triple"
            print("\nNo triples!")
            return templates.TemplateResponse('contextualization.html', context={
                'request': request,
                'highlevelgoal': highlevelgoal,
                'message': message,
                'goal_with_outputs': goal_with_outputs,
                'all_goal': all_goal,  # for the input (for autocompletion)
            })
    else:
        return RedirectResponse(f"/contextualization/{goal_with_outputs.id}", status_code=302)
