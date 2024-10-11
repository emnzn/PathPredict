from typing import Dict, Union

import numpy as np
import pandas as pd

from .patch import extract_coords

def save_embeddings(
    results: Dict[str, Union[str, np.ndarray]],
    save_dir: int
    ) -> None:

    """
    Saves the embeddings and coordinates of each patch as a parquet.

    Parameters
    ----------
    results: Dict[str, Union[str, np.ndarray]]
        A dictionary containing the coordinates and embeddings of each patch.
    """

    df = pd.DataFrame(results)
    df["processed_coords"] = df["coords"].map(lambda x: extract_coords(x))
    df = df.sort_values(by="processed_coords", key=lambda col: col.map(lambda x: (x[2], x[3], x[0], x[1])))
    df.to_parquet(save_dir, index=False)
