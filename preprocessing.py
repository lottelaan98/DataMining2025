#functies voor textclaining en vectorization
"""
preprocessing.py
----------------
Functies om de Ott et al. op_spam_v1.4 dataset in te lezen en handig op te slaan.

Belangrijk:
- We bewaren per review: text, label ('deceptive' of 'truthful'), y (int: 1=fake, 0=real),
  polarity ('negative' of 'positive'), fold (1–5), bestandsnaam en pad.
- Folds 1–4 zijn standaard trainingsfolds; fold 5 is standaard testfold (volgens de opdracht).
- We lezen standaard alleen 'negative' polarity in, omdat de opdracht daarop focust,
  maar je kunt 'positive' of beide meegeven als je wilt experimenteren.

"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Tuple, List, Literal, Optional
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import pandas as pd
import re

Polarity = Literal["negative", "positive"]
LabelStr = Literal["deceptive", "truthful"]


def load_op_spam(
    root: str | Path,
    polarities: Iterable[Polarity] = ("negative",),
    folds: Iterable[int] = (1, 2, 3, 4, 5),
    drop_duplicates: bool = True,
) -> pd.DataFrame:
    """
    Lees de op_spam_v1.4 dataset in als een DataFrame.

    Parameters
    ----------
    root : str | Path
        Pad naar de hoofdmap die de submappen 'negative_polarity' en/of 'positive_polarity' bevat.
        Dit is doorgaans: .../op_spam_v1.4
    polarities : Iterable[Polarity], default ("negative",)
        Welke polariteiten in te lezen: ("negative",), ("positive",) of ("negative","positive")
    folds : Iterable[int], default (1,2,3,4,5)
        Welke folds in te lezen (1..5)
    drop_duplicates : bool, default True
        Verwijder identieke teksten (veiligheidsnet).

    Returns
    -------
    pd.DataFrame met kolommen:
        - text (str): de ruwe reviewtekst
        - label (str): 'deceptive' of 'truthful'
        - y (int): 1 indien deceptive (fake), 0 indien truthful (real)
        - polarity (str): 'negative' of 'positive'
        - fold (int): 1..5
        - filename (str)
        - path (str): absoluut pad naar het .txt bestand
    """
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"Root pad bestaat niet: {root}")

    rows: List[dict] = []

    for polarity in polarities:
        pol_dir = root / f"{polarity}_polarity"
        if not pol_dir.exists():
            raise FileNotFoundError(f"Verwachte map niet gevonden: {pol_dir}")

        # Twee bronmappen per polariteit
        sources = [
            ("deceptive_from_MTurk", "deceptive", 1),
            ("truthful_from_Web", "truthful", 0),
        ]

        for src_dirname, label_str, y_val in sources:
            for fold in folds:
                fold_dir = pol_dir / src_dirname / f"fold{fold}"
                if not fold_dir.exists():
                    raise FileNotFoundError(f"Verwachte foldmap niet gevonden: {fold_dir}")

                for txt_path in sorted(fold_dir.glob("*.txt")):
                    try:
                        text = txt_path.read_text(encoding="utf-8").strip()
                    except UnicodeDecodeError:
                        # Fallback indien encoding afwijkt
                        text = txt_path.read_text(encoding="latin-1").strip()

                    rows.append(
                        dict(
                            text=text,
                            label=label_str,
                            y=int(y_val),
                            polarity=str(polarity),
                            fold=int(fold),
                            filename=txt_path.name,
                            path=str(txt_path.resolve()),
                        )
                    )

    df = pd.DataFrame(rows)

    #Willen we die duplicates echt verwijderen?
    if drop_duplicates and not df.empty:
        df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)

    # Zorg voor consistente dtypes
    df["label"] = df["label"].astype("category")
    df["polarity"] = df["polarity"].astype("category")
    df["fold"] = df["fold"].astype(int)
    df["y"] = df["y"].astype(int)

    return df


def split_by_fold(
    df: pd.DataFrame,
    train_folds: Iterable[int] = (1, 2, 3, 4),
    test_fold: int = 5,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits DataFrame in een train- en testdeel op basis van de foldnummers.
    Standaard: folds 1–4 -> train, fold 5 -> test (zoals de opdracht voorschrijft).

    Returns
    -------
    (train_df, test_df)
    """
    train_folds = set(int(f) for f in train_folds)
    test_fold = int(test_fold)

    train_df = df[df["fold"].isin(train_folds)].copy()
    test_df = df[df["fold"] == test_fold].copy()

    # Veiligheidschecks
    if train_df.empty:
        raise ValueError("Train-set is leeg. Controleer je train_folds.")
    if test_df.empty:
        raise ValueError("Test-set is leeg. Controleer je test_fold.")

    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)

def normalize_text(s: str) -> str:
    """
    Eenvoudige, taal-agnostische normalisatie:
    - lowercasing
    - cijfers en leestekens weg
    - spaties normaliseren
    """
    s = s.lower()
    s = re.sub(r"\d+", " ", s)
    s = re.sub(r"[^a-z\s]", " ", s)   # laat alleen a-z en spaties toe
    s = re.sub(r"\s+", " ", s).strip()
    return s


def normalize_series(texts: pd.Series) -> pd.Series:
    """Pas normalize_text toe op een pandas Series met tekst."""
    return texts.astype(str).map(normalize_text)

def make_vectorizer(
    use_tfidf: bool = False,
    ngram_range: tuple[int, int] = (1, 1),
    max_features: int | None = 5000,
    min_df: int | float = 2,
    max_df: int | float = 0.9,
    stop_words: str | None = "english",
    lowercase: bool = False,
):
    """
    Bouw een CountVectorizer of TfidfVectorizer met consistente defaults.
    - ngram_range=(1,1) -> unigrams; (1,2) -> unigrams + bigrams
    - zet lowercase=False als je normalize_text gebruikt (doen we standaard)
    """
    vec_cls = TfidfVectorizer if use_tfidf else CountVectorizer
    return vec_cls(
        ngram_range=ngram_range,
        max_features=max_features,
        min_df=min_df,
        max_df=max_df,
        stop_words=stop_words,
        lowercase=lowercase,   # we lowercasen al in normalize_text
    )


def vectorize_train_test(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    text_col: str = "text",
    y_col: str = "y",
    normalize: bool = True,
    **vectorizer_kwargs,
):
    """
    Maak X_train/X_test/y_train/y_test + de fitted vectorizer.
    Voorbeeld:
        Xtr, Xte, ytr, yte, vec = vectorize_train_test(
            train_df, test_df, ngram_range=(1,2), use_tfidf=False, max_features=5000
        )
    """
    Xtr_text = train_df[text_col].astype(str)
    Xte_text = test_df[text_col].astype(str)

    if normalize:
        Xtr_text = normalize_series(Xtr_text)
        Xte_text = normalize_series(Xte_text)

    vectorizer = make_vectorizer(**vectorizer_kwargs)
    X_train = vectorizer.fit_transform(Xtr_text)
    X_test = vectorizer.transform(Xte_text)
    y_train = train_df[y_col].values
    y_test = test_df[y_col].values

    return X_train, X_test, y_train, y_test, vectorizer


def get_feature_names(vectorizer) -> list[str]:
    """Compatibele manier om feature-namen op te halen."""
    # scikit-learn 1.0+:
    if hasattr(vectorizer, "get_feature_names_out"):
        return list(vectorizer.get_feature_names_out())
    # oudere versies:
    return list(vectorizer.get_feature_names())