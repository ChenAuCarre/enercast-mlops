"""Séparation chronologique des données d'entraînement et de validation.

Le jeu de test final n'est jamais utilisé pour choisir un modèle : il est
réservé à l'évaluation finale indépendante. La validation, elle, sert à
comparer le champion et le candidat lors d'un cycle de Continuous Training.
"""

import pandas as pd

# Fraction du pool d'entraînement réservée à la validation.
# La validation correspond toujours aux dernières observations du pool.
VALIDATION_FRACTION = 0.15


def split_train_validation(
    dataset: pd.DataFrame,
    validation_fraction: float = VALIDATION_FRACTION,
    date_column: str = "date",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Sépare un pool chronologique en entraînement puis validation.

    La validation est la fin de la période : elle est postérieure à toutes
    les observations d'entraînement, ce qui évite toute fuite du futur vers
    le passé. Le jeu de test final n'est pas concerné par cette fonction.
    """
    if not 0 < validation_fraction < 1:
        raise ValueError(
            "validation_fraction doit être strictement comprise entre 0 et 1."
        )

    if date_column not in dataset.columns:
        raise ValueError(f"Colonne de date absente : {date_column}")

    if len(dataset) < 2:
        raise ValueError("Le jeu de données doit contenir au moins deux lignes.")

    dates = pd.to_datetime(dataset[date_column], errors="raise")

    if dates.isna().any():
        raise ValueError("Le jeu de données contient des dates manquantes.")

    if not dates.is_monotonic_increasing:
        raise ValueError("Le jeu de données doit être trié par date croissante.")

    if dates.duplicated().any():
        raise ValueError("Le jeu de données contient des dates dupliquées.")

    # Arrondi vers le bas, avec au moins une ligne de validation et une
    # ligne d'entraînement pour que les deux jeux restent exploitables.
    validation_rows = int(len(dataset) * validation_fraction)
    validation_rows = max(1, min(validation_rows, len(dataset) - 1))

    train = dataset.iloc[:-validation_rows].copy()
    validation = dataset.iloc[-validation_rows:].copy()

    if (
        pd.to_datetime(train[date_column]).max()
        >= pd.to_datetime(validation[date_column]).min()
    ):
        raise ValueError(
            "La séparation entraînement/validation n'est pas chronologique."
        )

    return train, validation


def load_train_validation(
    train_path,
    validation_fraction: float = VALIDATION_FRACTION,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Lit le pool d'entraînement et sépare entraînement / validation.

    Le jeu de test final n'est volontairement pas lu ici : il ne doit jamais
    être consulté pendant l'entraînement ni pendant un cycle de CT.
    """
    train_pool = pd.read_csv(train_path)

    return split_train_validation(
        train_pool,
        validation_fraction=validation_fraction,
    )


def load_final_test(test_path) -> pd.DataFrame:
    """Lit le jeu de test final.

    Cette fonction est réservée à l'évaluation finale indépendante, exécutée
    après la sélection du modèle. Elle ne doit pas être appelée par le chemin
    d'entraînement, de comparaison ou de promotion.
    """
    return pd.read_csv(test_path)
