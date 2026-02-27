from pydantic import BaseModel, Field
from typing import List, Optional

class OptimizationRequest(BaseModel):
    """
    Schema for the incoming solar cell optimization request.
    """
    materials: List[str] = Field(
        ..., 
        min_length=1, 
        max_length=5, 
        examples=[["PbS", "CdSe"]],
        description="List of semiconductor materials for the tandem architecture."
    )
    
    operating_temperature: float = Field(
        298.15, 
        ge=0.0, 
        le=500.0, 
        description="Operating temperature in Kelvin (standard is 298.15K)."
    )
    
    population_size: int = Field(
        100, 
        gt=0, 
        le=1000, 
        description="Number of candidate solutions in the Genetic Algorithm."
    )
    
    max_iterations: int = Field(
        50, 
        gt=0, 
        le=500, 
        description="Number of generations for the evolutionary process."
    )
    
    crossover_alpha: float = Field(
        0.5, 
        ge=0.0, 
        le=1.0, 
        description="Weight for arithmetic crossover in tensor evolution."
    )
    
    mutation_strength: float = Field(
        0.1, 
        ge=0.0, 
        le=1.0, 
        description="Standard deviation for Gaussian mutation."
    )

class OptimizationResponse(BaseModel):
    """
    Schema for the result returned to the researcher.
    """
    simulation_id: str
    status: str = "COMPLETED"
    optimal_radii_nm: List[float]
    absorption_spectrum: List[List]
    projected_pce: float
    fitness_history: List[float]
    computation_time_ms: float