import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent
SRC = Path(__file__).resolve().parent / "src"
sys.path.append(str(ROOT))
sys.path.append(str(SRC))


from src.df_deploy.predicting.main import main


if __name__ == "__main__":
    main()