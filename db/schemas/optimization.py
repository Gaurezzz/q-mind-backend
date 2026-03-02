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

    kappa: float = Field(
        0.5,
        ge=0.0,
        le=1.0,
        description="Penalty coefficient for current mismatch in the fitness function. Higher values increase the penalty for current mismatch in tandem architectures, driving the optimization towards better current matching across layers."
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
    optimal_radii_nm: List[float] = Field(
        description="Optimal QD radii for each layer in nanometers. Shape: (num_layers,)",
    )
    projected_pce: float = Field(
        description="Projected Power Conversion Efficiency (PCE) of the optimal configuration.",
    )
    fitness_history: List[float] = Field(
        description="History of fitness values across generations. Shape: (iterations,)",
    )
    pce_history: List[float] = Field(
        description="History of projected PCE values across generations. Shape: (iterations,)",
    )
    avg_fitness_history: List[float] = Field(
        description="History of mean population fitness values across generations. Shape: (iterations,)",
    )
    computation_time_ms: float = Field(
        description="Total computation time for the optimization process in milliseconds.",
    )
    materials: List[str] = Field(
        description="List of materials used in the optimization."
    )
    bandgaps_eV: List[float] = Field(
        description="Bandgap energies in eV for the optimal configuration."
    )
    wavelengths_nm: List[float] = Field(
        description="Wavelength grid used in the optimization in nm."
    )
    absorption_spectrum: List[List[float]] = Field(
        description="Absorption spectrum of the optimal configuration. Shape: (num_layers, num_wavelengths)."
    )
    current_mismatch_index: float = Field(
        description="Normalized current mismatch index for tandem architectures. A value of 0 indicates perfect current matching across all layers. Higher values represent energy losses due to the 'bottleneck effect,' where the layer with the lowest photocurrent limits the total device performance."
    )
    photon_harvesting_efficiency: List[float] = Field(
        description="Photon harvesting efficiency for each layer at the optimal configuration. Shape: (num_layers,)"
    )
    generations_to_convergence: int = Field(
        description="Number of generations required to reach the optimal solution or convergence criteria."
    )


