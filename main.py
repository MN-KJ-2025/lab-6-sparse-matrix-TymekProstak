# =================================  TESTY  ===================================
# Testy do tego pliku zostały podzielone na dwie kategorie:
#
#  1. `..._invalid_input`:
#     - Sprawdzające poprawną obsługę nieprawidłowych danych wejściowych.
#
#  2. `..._correct_solution`:
#     - Weryfikujące poprawność wyników dla prawidłowych danych wejściowych.
# =============================================================================
import numpy as np
import scipy as sp

def is_diagonally_dominant(A: np.ndarray | sp.sparse.csc_array) -> bool | None:
    if not isinstance(A, (np.ndarray, sp.sparse.csc_array)):
        return None

    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        return None

    if sp.sparse.issparse(A):
        A = A.toarray()

    diag = np.abs(np.diagonal(A))
    off_diag_sum = np.sum(np.abs(A), axis=1) - diag

    return bool(np.all(diag > off_diag_sum))



def residual_norm(A: np.ndarray, x: np.ndarray, b: np.ndarray) -> float | None:
    """Funkcja obliczająca normę residuum dla równania postaci: 
    Ax = b.

    Args:
        A (np.ndarray): Macierz A (m,n) zawierająca współczynniki równania.
        x (np.ndarray): Wektor x (n,) zawierający rozwiązania równania.
        b (np.ndarray): Wektor b (m,) zawierający współczynniki po prawej 
            stronie równania.
    
    Returns:
        (float): Wartość normy residuum dla podanych parametrów.
        Jeżeli dane wejściowe są niepoprawne funkcja zwraca `None`.
    """

    # Sprawdzenie typów
    if not isinstance(A, np.ndarray) or not isinstance(x, np.ndarray) or not isinstance(b, np.ndarray):
        return None

    # Sprawdzenie wymiarów
    if A.ndim != 2 or x.ndim != 1 or b.ndim != 1:
        return None
    if A.shape[0] != b.shape[0] or A.shape[1] != x.shape[0]:
        return None

    # Obliczenie normy residuum ‖Ax - b‖₂
    r = A @ x - b
    return np.linalg.norm(r, 2)
