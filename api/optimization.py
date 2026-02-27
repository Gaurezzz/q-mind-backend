import time
import uuid
import numpy as np
import mindspore as ms
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from db import models
from db.config import get_db
from db.schemas.optimization import OptimizationRequest, OptimizationResponse
from api.deps import get_current_user
from logic.SolarOptimizationManager import SolarOptimizationManager

CSV_PATH = "db/materials.csv"

router = APIRouter(
    prefix="/optimization",
    tags=["optimization"]
)

@router.post("/run", response_model=OptimizationResponse, status_code=status.HTTP_200_OK)
def run_optimization(
    request: OptimizationRequest,
    db: Session = Depends(get_db),
    current_user: models.User = Depends(get_current_user)
):
    """
    run_optimization: Executes a genetic algorithm-based solar cell optimization.

    Args:
        request (OptimizationRequest): Optimization parameters including materials,population size, iterations, and GA hyperparameters.
        db (Session): Database session.
        current_user (User): Authenticated user.

    Returns:
        OptimizationResponse: Optimal QD radii, projected PCE, and fitness history.
    """
    manager = SolarOptimizationManager(csv_path=CSV_PATH)

    available_materials = list(manager.catalog.keys())
    invalid = [m for m in request.materials if m not in available_materials]
    if invalid:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Unknown materials: {invalid}. Available: {available_materials}"
        )

    user_params = {
        'materials':   request.materials,
        'pop_size':    request.population_size,
        'alpha':       request.crossover_alpha,
        'mutation':    request.mutation_strength,
        'iterations':  request.max_iterations,
        'temp':        ms.Tensor([request.operating_temperature], ms.float32),
        'wavelength':  ms.Tensor(np.arange(280, 2000, 20, dtype=np.float32), ms.float32),
    }

    start = time.perf_counter()
    fitness_history, best_radii, absorption_spectrum = manager.run_study(user_params)  # type: ignore
    elapsed_ms = (time.perf_counter() - start) * 1000

    optimal_radii = best_radii.asnumpy().tolist() # type: ignore
    projected_pce = fitness_history[-1] if fitness_history else 0.0

    return OptimizationResponse(
        simulation_id=str(uuid.uuid4()),
        status="COMPLETED",
        optimal_radii_nm=optimal_radii,
        projected_pce=projected_pce,
        fitness_history=fitness_history,
        absorption_spectrum=absorption_spectrum,
        computation_time_ms=round(elapsed_ms, 2)
    )
