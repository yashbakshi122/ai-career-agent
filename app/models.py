from pydantic import BaseModel
from typing import List

class CareerAdvice(BaseModel):
    strengths: List[str]
    skill_gaps: List[str]
    learning_plan: List[str]
    interview_tips: List[str]

