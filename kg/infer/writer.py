"""Hipotez yazma servisi. Çıkarım katmanının grafa yazabildiği TEK yol budur ve yalnızca
HYPOTHESIS kenarı yazabilir; kanıt kenarlarına erişimi yoktur. Ajan bu servise doğrudan
erişemez — ajan aday döner, yazma kararı bu servisten geçer."""
from __future__ import annotations

from ..graph import GraphStore
from ..schema import Hypothesis


class HypothesisWriter:
    def __init__(self, store: GraphStore):
        self._upsert = store.upsert_hypothesis  # depodan yalnızca bu metot tutulur

    def write(self, h: Hypothesis) -> None:
        if not h.path:
            raise ValueError("yolu (path) olmayan hipotez yazılamaz")
        if any(step.get("predicate") == "HYPOTHESIS" for step in h.path):
            raise ValueError("bir hipotez başka bir hipotezin kanıtı olarak kullanılamaz")
        self._upsert(h)
