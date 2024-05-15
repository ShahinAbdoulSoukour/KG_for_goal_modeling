import json
from sqlalchemy import Column, Integer, String, Float, Boolean
from database import Base
from sqlalchemy.dialects.postgresql import ARRAY, VARCHAR

#class Unique_Entailed_Triples(Base):
#    __tablename__ = "unique_entailed_triples"
#    id = Column(Integer, primary_key=True)
#    triple = Column(ARRAY(String(100)))

class Anchor_Points(Base):
    __tablename__ = "anchor_points"

    id = Column(Integer, primary_key=True)
    triple = Column(String(100))
    triple_serialized = Column(String(100))  # Storing the serialized list as a string
    subject = Column(String(100))
    predicate = Column(String(100))
    object = Column(String(100))
    goal = Column(String(100))

    def set_triple_serialized(self, triple_list):
        self.triple_serialized = json.dumps(triple_list)

    #def get_triple_serialized(self):
    #    return json.loads(self.triple_serialized) if self.triple_serialized else []


class Entailment_Results(Base):
    __tablename__ = "entailment_results"

    id = Column(Integer, primary_key=True)
    premise = Column(String(100))
    hypothesis = Column(String(100))
    premise_serialized = Column(String(100))
    subject = Column(String(100))
    predicate = Column(String(100))
    object = Column(String(100))
    entailment = Column(Float)
    neutral = Column(Float)
    contradiction = Column(Float)
    nli_label = Column(String(15))

    def set_premise_serialized(self, triple_list):
        self.premise_serialized = json.dumps(triple_list)

    def get_premise_serialized(self):
        return json.loads(self.premise_serialized)