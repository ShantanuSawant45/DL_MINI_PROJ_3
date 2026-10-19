from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.resolve()
RESULTS_DIR  = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

DATASETS: list[str] = ["mnist", "fashion_mnist"]

CODE_SIZES: list[int] = [2, 8, 16, 32, 64, 128, 256]

K_VALUES: list[int] = [10, 20, 30]

N_RUNS: int = 5

BATCH_SIZE: int  = 64
LR: float        = 8e-4
EPOCHS: int      = 30
DROPOUT_PAPER_VALUE: float = 0.8

N_EXPONENTS: list[float] = [0, 0.01, 0.05, 0.1, 0.5, 1, 2, 4, 8]

BEST_CODE_SIZE: dict[str, int] = {
    "mnist":         16,
    "fashion_mnist": 16,
}

GLOBAL_SEED: int = 42
IMAGE_SHAPE: tuple[int, int, int] = (28, 28, 1)
