from pydantic import BaseModel, Field, model_validator
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
        gt=1, 
        le=1000, 
        description="Number of candidate solutions in the Genetic Algorithm."
    )
    
    max_iterations: int = Field(
        50, 
        gt=1, 
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

    wavelength_input_csv: bool = Field(
        False,
        description="If true, the optimization will use a custom wavelength grid provided in the request instead of the default AM1.5G spectrum."
    ) 

    wavelength_left_bound: Optional[float] = Field(
        None,
        ge=280.0,
        le=2499.9,
        description="Left bound of the wavelength range in nm (required if wavelength_input_csv is false)."
    )

    wavelength_right_bound: Optional[float] = Field(
        None,
        ge=280.1,
        le=2500.0,
        description="Right bound of the wavelength range in nm (required if wavelength_input_csv is false)."
    )

    wavelength_step: Optional[float] = Field(
        None,
        ge=0.1,
        le=100.0,
        description="Step size for the wavelength grid in nm (used if wavelength_input_csv is false)."
    )

    @model_validator(mode="after")
    def check_wavelength_requirements(self) -> 'OptimizationRequest':

        verify_wavelengths = (self.wavelength_input_csv == (self.wavelength_left_bound is None)) and (self.wavelength_input_csv == (self.wavelength_right_bound is None)) and (self.wavelength_input_csv == (self.wavelength_step is None))

        if not verify_wavelengths:
            raise ValueError("When wavelength_input_csv is false, wavelength_left_bound, wavelength_right_bound, and wavelength_step must all be provided. When wavelength_input_csv is true, these fields must all be None.")


        if not self.wavelength_input_csv and self.wavelength_right_bound <= self.wavelength_left_bound:
            raise ValueError("wavelength_right_bound must be greater than wavelength_left_bound.")  
        
        return self


class OptimizationResponse(BaseModel):
    """
    Schema for the result returned to the researcher.
    """
    status: str = "COMPLETED"
    optimal_radii_nm: List[float]
    absorption_spectrum: List[List]
    projected_pce: float
    fitness_history: List[float]
    pce_history: List[float]
    computation_time_ms: float