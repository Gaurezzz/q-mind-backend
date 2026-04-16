import time
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from db import models
from db.config import get_db
from db.schemas.optimization import OptimizationRequest, OptimizationResponse
from api.deps import get_current_user
from modelarts_worker.logic.DataAnalyzer import DataAnalyzer
from modelarts_worker.mindspore_config import get_logger

logger = get_logger(__name__)

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
        request (OptimizationRequest): Optimization parameters including materials,
            population size, iterations, GA hyperparameters, and wavelength range.
        db (Session): Database session.
        current_user (User): Authenticated user.

    Returns:
        OptimizationResponse: Optimal QD radii, projected PCE, fitness/PCE histories,
            bandgaps, current mismatch index, photon harvesting efficiency, and
            convergence generation.
    """
    analyzer = DataAnalyzer(csv_path=CSV_PATH, kappa=request.kappa)

    invalid = analyzer.validate_materials(request.materials)
    if invalid:
        logger.warning(
            "Unknown materials requested by user '%s': %s",
            current_user.username if hasattr(current_user, 'username') else current_user.id,
            invalid,
        )
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Unknown materials: {invalid}. Available: {analyzer.available_materials}"
        )

    if request.wavelength_input_csv:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail="CSV wavelength input is not yet supported. Please use the automatic range mode."
        )

    logger.info(
        "Optimization request | user=%s  materials=%s  pop=%d  iters=%d",
        current_user.username if hasattr(current_user, 'username') else current_user.id,
        request.materials,
        request.population_size,
        request.max_iterations,
    )

    start = time.perf_counter()
    try:
        result = analyzer.analyze(request)
    except Exception:
        logger.exception(
            "Unhandled error during optimization | materials=%s", request.materials
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Optimization failed. Check server logs for details.",
        )
    elapsed_ms = (time.perf_counter() - start) * 1000

    logger.info(
        "Optimization DONE | materials=%s  time=%.0f ms  pce=%.4f%%",
        request.materials,
        elapsed_ms,
        result.get("projected_pce", 0) * 100,
    )

    return OptimizationResponse(
        status="COMPLETED",
        computation_time_ms=round(elapsed_ms, 2),
        **result
    )

