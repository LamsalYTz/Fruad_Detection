from pydantic import BaseModel, Field
from typing import List

class fraud_response(BaseModel):
    is_fraud : bool
    fraud_label : str
    confidence : float
    risk_level : str
    recommendation : str
    amount : float