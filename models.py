import json
from sqlalchemy import create_engine, Column, Integer, String, Table, ForeignKey, Float, Boolean
from sqlalchemy.orm import declarative_base, relationship
from database import Base
from sqlalchemy.dialects.postgresql import ARRAY, VARCHAR

#class Unique_Entailed_Triples(Base):
#    __tablename__ = "unique_entailed_triples"
#    id = Column(Integer, primary_key=True)
#    triple = Column(ARRAY(String(100)))

#class Anchor_Points(Base):
#    __tablename__ = "anchor_points"

#    id = Column(Integer, primary_key=True)
#    triple = Column(String(100))
#    triple_serialized = Column(String(100))  # Storing the serialized list as a string
#    subject = Column(String(100))
#    predicate = Column(String(100))
#    object = Column(String(100))
#    goal = Column(String(100))

#    def set_triple_serialized(self, triple_list):
#        self.triple_serialized = json.dumps(triple_list)

#    def get_triple_serialized(self):
#        return json.loads(self.triple_serialized) if self.triple_serialized else []


# Hierarchy table to manage parent-child relationships between goals
class Hierarchy(Base):
    __tablename__ = "hierarchy"
    id = Column(Integer, primary_key=True)
    high_level_goal_id = Column(Integer, ForeignKey('goal.id'))
    subgoal_id = Column(Integer, ForeignKey('goal.id'))

    high_level_goal = relationship("Goal", foreign_keys=[high_level_goal_id], back_populates="high_level_hierarchies")
    subgoal = relationship("Goal", foreign_keys=[subgoal_id], back_populates="subgoal_hierarchies")



class Goal(Base):
    __tablename__ = "goal"

    id = Column(Integer, primary_key=True)
    goal_name = Column(String(100))

    outputs = relationship("Outputs", back_populates="goal") # one-to-many relationship`

    triple_filtered = relationship("Triple_Filtered", back_populates="goal",
                                   foreign_keys="[Triple_Filtered.high_level_goal_id]")
    triple_filtered_subgoals = relationship("Triple_Filtered", back_populates="subgoal",
                                            foreign_keys="[Triple_Filtered.subgoal_id]")

    high_level_hierarchies = relationship("Hierarchy", foreign_keys="[Hierarchy.high_level_goal_id]",
                                          back_populates="high_level_goal")
    subgoal_hierarchies = relationship("Hierarchy", foreign_keys="[Hierarchy.subgoal_id]", back_populates="subgoal")

    #parent_goal = relationship("Goal", remote_side=[id], back_populates="subgoals")
    #subgoals = relationship("Goal", back_populates="parent_goal", cascade="all, delete-orphan")

    # Self-referential many-to-many relationship for goal hierarchy
    #parents = relationship(
    #    "Goal",
    #    secondary="hierarchy",
    #    primaryjoin=id == Hierarchy.child_id,
    #    secondaryjoin=id == Hierarchy.parent_id,
    #    back_populates="children"
    #)
    #children = relationship(
    #    "Goal",
    #    secondary="hierarchy",
    #    primaryjoin=id == Hierarchy.parent_id,
    #    secondaryjoin=id == Hierarchy.child_id,
    #    back_populates="parents"
    #)



class Outputs(Base):
    __tablename__ = "outputs"

    id = Column(Integer, primary_key=True)
    generated_text = Column(String(100))
    entailed_triple = Column(String(100))
    goal_id = Column(Integer, ForeignKey('goal.id'))

    goal = relationship("Goal", back_populates="outputs") # many-to-one relationship
    #triple_filtered = relationship("Triple_Filtered", back_populates="outputs")

    def set_entailed_triple(self, triple_list):
        self.entailed_triple = json.dumps(triple_list)

    def get_entailed_triples(self):
        return json.loads(self.entailed_triple)





class Triple_Filtered(Base):
    __tablename__ = "triple_filtered"

    id = Column(Integer, primary_key=True)
    #output_id = Column(Integer, ForeignKey('outputs.id'))
    ##goal_id = Column(Integer, ForeignKey('goal.id'))
    subgoal_id = Column(Integer, ForeignKey('goal.id'))
    ##level = Column(Integer)
    ##triple_filtered = Column(String(100))
    triple_filtered_from_hlg = Column(String(100)) # triple used to create the subgoal
    high_level_goal_id = Column(Integer, ForeignKey('goal.id'))

    #goal = relationship("Goal", back_populates="triple_filtered")
    #outputs = relationship("Outputs", back_populates="triple_filtered")

    goal = relationship("Goal", back_populates="triple_filtered", foreign_keys=[high_level_goal_id])
    subgoal = relationship("Goal", back_populates="triple_filtered_subgoals", foreign_keys=[subgoal_id])

    def set_entailed_triple(self, triple_list):
        self.triple_filtered_from_hlg = json.dumps(triple_list)

    def get_entailed_triples(self):
        return json.loads(self.triple_filtered_from_hlg)

