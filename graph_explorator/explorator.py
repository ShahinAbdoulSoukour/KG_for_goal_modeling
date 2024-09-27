import pandas as pd

from utils.functions import get_neighbors, test_entailment_api


def graph_explorator(df, goal, graph, model_nli_name):
    entailed_triples_df = pd.DataFrame(columns=["GOAL_TYPE", "SUBGOALS", "SUBGOALS_SERIALIZED", "SCORE", "NLI_LABEL"])

    for _, row in df[["GOAL_TYPE", "PREMISE", "HYPOTHESIS", "PREMISE_SERIALIZED", "ENTAILMENT", "NLI_LABEL"]].iterrows():
        # Entailment score
        entailment_score = row["ENTAILMENT"]

        # Check if there is an entailment
        if row['NLI_LABEL'] == "ENTAILMENT":
            entailed_triples_dict = {
                "GOAL_TYPE": row['GOAL_TYPE'],
                "SUBGOALS": row["PREMISE"],
                "SUBGOALS_SERIALIZED": row["PREMISE_SERIALIZED"],
                "SCORE": entailment_score,
                "NLI_LABEL": row["NLI_LABEL"]
            }

            # update the dataframe
            entailed_triples_df.loc[len(entailed_triples_df)] = entailed_triples_dict
        else: # If no entailment
            # extract triple neighbors
            neighbor_triples = get_neighbors(row["PREMISE"], row["PREMISE_SERIALIZED"], graph)

            # concatenate triples
            concatenated_triples_dict = {
                "GOAL_TYPE": row["GOAL_TYPE"], #
                "PREMISE": neighbor_triples["TRIPLE_NEIGHBOR"] + ". " + neighbor_triples['TRIPLE'],
                "HYPOTHESIS": goal,
                "PREMISE_SERIALIZED": neighbor_triples["TRIPLE_NEIGHBOR_SERIALIZED"] + neighbor_triples['TRIPLE_SERIALIZED']
            }
            concatenated_triples_df = pd.DataFrame(concatenated_triples_dict)

            # --> test the entailment
            entailment_concatenate_triples_result = test_entailment_api(concatenated_triples_df, model_nli_name)

            print("\nENTAILMENT TEST RESULTS (CONCATENATED TRIPLES):")
            print(entailment_concatenate_triples_result.to_string())

            # extract the highest score (entailment)
            entailment_concatenate_triples_best_score = entailment_concatenate_triples_result.head(1)

            # check if the label is "ENTAILMENT"
            contains_entailment = any(entailment_concatenate_triples_best_score['NLI_LABEL'] == 'ENTAILMENT')

            if contains_entailment:
                # for index, row in entailment_concatenate_triples_best_score.iterrows():
                # if all(all(elem not in sublist for elem in row['PREMISE_SERIALIZED']) for sublist in entailed_triples_df['SUBGOALS_SERIALIZED']):
                entailment_concatenate_triples_best_score = entailment_concatenate_triples_best_score[
                    ['GOAL_TYPE', 'PREMISE', 'PREMISE_SERIALIZED', 'ENTAILMENT', 'NLI_LABEL']]
                entailment_concatenate_triples_best_score.rename(
                    columns={'PREMISE': 'SUBGOALS', 'PREMISE_SERIALIZED': 'SUBGOALS_SERIALIZED', 'ENTAILMENT': 'SCORE'},
                    inplace=True)

                # update the dataframe
                entailed_triples_df = pd.concat([entailed_triples_df, entailment_concatenate_triples_best_score])
            else:
                while True:
                    current_score = entailment_concatenate_triples_result.iloc[0]['ENTAILMENT']

                    # check the evolution of the entailment score
                    if current_score > entailment_score:
                        # extract the triple neighbors
                        neighbor_triples = get_neighbors(entailment_concatenate_triples_best_score.iloc[0]["PREMISE"],
                                                         entailment_concatenate_triples_best_score.iloc[0]["PREMISE_SERIALIZED"],
                                                         graph)

                        if not neighbor_triples.empty:
                            # concatenate triples
                            concatenated_triples_dict = {
                                "GOAL_TYPE": row["GOAL_TYPE"],
                                "PREMISE": neighbor_triples["TRIPLE_NEIGHBOR"] + ". " + neighbor_triples['TRIPLE'],
                                "HYPOTHESIS": goal,
                                "PREMISE_SERIALIZED": neighbor_triples["TRIPLE_NEIGHBOR_SERIALIZED"] + neighbor_triples['TRIPLE_SERIALIZED']
                            }
                            concatenated_triples_df = pd.DataFrame(concatenated_triples_dict)

                            # --> test the entailment
                            entailment_concatenate_triples_result = test_entailment_api(concatenated_triples_df, model_nli_name)

                            print("\nENTAILMENT TEST RESULTS (CONCATENATED TRIPLES):")
                            print(entailment_concatenate_triples_result.to_string())

                            # update the previous score
                            entailment_score = current_score

                            ###
                            if any(entailment_concatenate_triples_result['NLI_LABEL'] == 'ENTAILMENT'):
                                entailment_concatenate_triples_result = entailment_concatenate_triples_result.loc[
                                    entailment_concatenate_triples_result['NLI_LABEL'] == 'ENTAILMENT'][
                                    ['GOAL_TYPE', 'PREMISE', 'PREMISE_SERIALIZED', 'ENTAILMENT', 'NLI_LABEL']]
                                entailment_concatenate_triples_result.rename(
                                    columns={'PREMISE': 'SUBGOALS', 'PREMISE_SERIALIZED': 'SUBGOALS_SERIALIZED',
                                             'ENTAILMENT': 'SCORE'}, inplace=True)

                                # update the dataframe
                                entailed_triples_df = pd.concat([entailed_triples_df, entailment_concatenate_triples_result])

                                break
                            ###
                        else:
                            #print("STOP THE EXPLORATION, MOVE TO THE NEXT...")
                            break
                    else:
                        #print("STOP THE EXPLORATION, MOVE TO THE NEXT...")
                        break

    print("\n=>ENTAILED TRIPLES:")
    print(entailed_triples_df.sort_values(['SCORE'], ascending=[False]).to_string())
    return entailed_triples_df