# retrievers/base.py

from abc import ABC, abstractmethod
from typing import List, Dict, Any


class BaseRetriever(ABC):
    """
    Abstract base class for all candidate retrievers.
    """

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def retrieve(
        self,
        text: str,
        context: str | None = None,
        top_k: int = 20,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve candidate ontology concepts for a given text.

        Returns a list of dicts with at least:
        {
            "concept_id": str,
            "concept_name": str,
            "score": float,
            "source": retriever_name,
            ... optional metadata ...
        }
        """
        raise NotImplementedError
