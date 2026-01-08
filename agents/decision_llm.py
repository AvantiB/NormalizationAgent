import json, requests, re
from typing import Any, Dict, List


class DecisionAgentLLM:
    def __init__(self, model_name: str, api_url: str):
        self.model = model_name
        self.url = api_url

    # --------------------------------------------------
    # Robust JSON parsing
    # --------------------------------------------------
    def _safe_json_loads(self, text: str) -> Dict[str, Any] | None:
        if not text or not text.strip():
            return None

        # Remove markdown fences
        text = text.strip()
        # Remove special unused tokens (model artifacts)
        text = re.sub(r"<unused\d+>", "", text)
        # Remove markdown fences
        text = text.replace("```json", "").replace("```", "").strip()

        # Try direct parse
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Fallback: extract first JSON object
        try:
            matches = re.findall(r"\{[\s\S]*?\}", text)
            if not matches:
                return None
            return json.loads(matches[-1])
        except json.JSONDecodeError:
            return None


    def decide(self, entity: Dict[str, Any], candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Decide the best concept for a SINGLE entity.

        Returns:
        {
            "status": "confident" | "uncertain" | "hard_fail",
            "concept_id": str | None,
            "concept_name": str | None,
            "hierarchy": str | None,
            "confidence": float,
            "source": str
        }
        """

        if not candidates:
            return {
                "status": "hard_fail",
                "concept_id": None,
                "concept_name": None,
                "hierarchy": None,
                "confidence": 0.0,
                "source": "no_candidates"
            }
        
        # Track allowed concept IDs (guardrail)
        allowed_ids = {c["concept_id"] for c in candidates}

        system_prompt = (
            "You are a careful clinical terminology normalization assistant.\n"
            "You must select the single best matching concept from the provided candidates by carefully consider the entity within the provided clinical note context.\n"
            "Rules:\n"
            "- Use ONLY the provided candidates.\n"
            "- Do NOT infer details not present in the entity/context text.\n"
            "- Select the concept without extra modifiers or additional information **unless those modifiers are explicitly mentioned in the text**. For example, if the entity is 'rigors' and the candidates are 'Rigor (finding)' and 'Shivering or rigors (finding)', select 'Rigor (finding)'.\n" 
            "- Select concept with specifications when explicitly mentioned in the entity. For example, if the entity is 'colonic diverticula', you must select the concept that specifically mentions 'Divertoculosis of **Colon** (finding)' \n"
            "- Select the concept with appropriate 'hierarchy' or semantic type based on the entity context.\n"
            "- If none are acceptable, return null and low confidence. Do not guess.\n\n"
            "Do not include reasoning, analysis, or explanations in your response. Do NOT include markdown.\n"
            "**Return ONLY the final JSON object and nothing else.**\n"
        )

        user_prompt = (
            f"- Entity text (verbatim): \"{entity['verbatim']}\"\n"
            f"- Entity text (abbreviation expanded): \"{entity['expanded']}\"\n"
            f"- Clinical note context: \"{entity['context']}\"\n\n"
            "Retrieved candidate concepts:\n"
            + json.dumps(candidates, ensure_ascii=False, indent=2)
            + "\n\n"
            "Return STRICTLY ONLY valid JSON and nothing else:\n"
            "{\n"
            "  \"concept_id\": \"...\" or null,\n"
            "  \"concept_name\": \"...\" or null,\n"
            "  \"hierarchy\": \"...\" or null,\n"
            "  \"confidence\": float between 0 and 1\n"
            "}"
        )

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0,
            "top_p": 0.9,
            "max_tokens": 4098,
        }

        try:
            resp = requests.post(self.url, json=payload)
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"]
            # print(content)
        except Exception as e:
            print("LLM error:", e)
            return {
                    "status": "hard_fail",
                    "concept_id": None,
                    "concept_name": None,
                    "hierarchy": None,
                    "confidence": 0.0,
                    "source": "llm_error"
                }

        parsed = self._safe_json_loads(content)

        if not parsed:
            print("LLM JSON parse failure")
            print("RAW OUTPUT:", content)
            return {
                    "status": "uncertain",
                    "concept_id": None,
                    "concept_name": None,
                    "hierarchy": None,
                    "confidence": 0.0,
                    "source": "parse_failure",
                }

        # Guardrail: ensure concept_id is from candidates
        cid = parsed.get("concept_id")
        if cid is not None and cid not in allowed_ids:
            print("LLM hallucinated concept_id:", cid)
            return {
                    "status": "uncertain",
                    "concept_id": None,
                    "concept_name": None,
                    "hierarchy": None,
                    "confidence": 0.0,
                    "source": "llm_hallucination",
                }

        parsed["status"] = "confident"
        parsed["source"] = "llm_decision"
        return parsed