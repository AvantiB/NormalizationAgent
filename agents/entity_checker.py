import json, requests, re
from typing import Any, Dict, List, Optional

class EntityCheckerAgent:
    def __init__(self, model_name: str, api_url: str):
        self.model = model_name
        self.url = api_url

    # --------------------------------------------------
    # Robust JSON parsing
    # --------------------------------------------------
    def _safe_json_loads(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Handles cases like:
        - extra chain-of-thought
        - <unused94> tokens
        - markdown fences
        - stray leading/trailing text

        Strategy:
        1) If there's a ```json ... ``` block, parse inside it.
        2) Else parse the last {...} object found in the string.
        """
        if not text or not text.strip():
            return None

        t = text.strip()

        # 1) Prefer fenced JSON block if present
        fence = re.search(r"```json\s*([\s\S]*?)\s*```", t, flags=re.IGNORECASE)
        if fence:
            candidate = fence.group(1).strip()
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                pass

        # Remove common fences if model forgets the "json" label
        t = t.replace("```json", "").replace("```", "").strip()

        # 2) Fallback: try to parse the LAST JSON object in the text
        objs = re.findall(r"\{[\s\S]*\}", t)
        if not objs:
            return None

        # Try from last to first (often the last block is the actual answer)
        for obj in reversed(objs):
            try:
                return json.loads(obj)
            except json.JSONDecodeError:
                continue

        return None

    def _call_llm(self, system_prompt: str, user_prompt: str) -> Optional[str]:
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
            return resp.json()["choices"][0]["message"]["content"]
        except Exception as e:
            print("LLM error:", e)
            return None

    def run(self, entity_text: str, context: str) -> dict:
        system_prompt = f"""
            You are a clinical entity normalization assistant.

            Your task:
            1. Decide whether the entity text is ambiguous, abbreviated, or misleading GIVEN the context.
            2. If yes, produce ONE normalized surface form likely to exist in SNOMED-CT labels.
            3. Optionally suggest the most likely SNOMED semantic category.
            4. Do NOT guess SNOMED-CT ontology IDs.
            5. Do NOT generate multiple rewrites.

            SNOMED semantic categories under consideration:
            - finding: An observation, sign, or symptom.
            - procedure: A purposeful action or healthcare intervention.
            - disorder: A specific pathological process or clinical condition.
            - body structure: Normal anatomical parts, organs, or regions.
            - morphologic abnormality: Structural changes caused by disease or injury.
            - regime/therapy: A structured plan or set of activities for treatment.
            - cell structure: Microscopic components or organelles within a cell.
            """

        user_prompt = f"""
            Input:

            Entity:
            "{entity_text}"

            Context:
            "{context}"

            Return JSON ONLY in this format:
            {{
            "needs_rewrite": true | false,
            "normalized_entity": "... or null",
            "semantic_tag_hint": "... or null",
            "reason": "..."
            }}
            """
        content = self._call_llm(system_prompt, user_prompt)
        if not content:
            return {
                "needs_rewrite": False,
                "normalized_entity": None,
                "semantic_tag_hint": None,
                "reason": "LLM_error"
            }
        parsed = self._safe_json_loads(content)
        if not parsed:
            print("LLM JSON parse failure (select)")
            print("RAW OUTPUT:", content)
            return {
                "needs_rewrite": False,
                "normalized_entity": None,
                "semantic_tag_hint": None,
                "reason": "parse_failure"
            }
        if not parsed.get("needs_rewrite"):
            return {
                "needs_rewrite": False,
                "normalized_entity": None,
                "semantic_tag_hint": None,
                "reason": parsed.get("reason", "")
            }
        
        return {
            "needs_rewrite": True,
            "normalized_entity": parsed.get("normalized_entity"),
            "semantic_tag_hint": parsed.get("semantic_tag_hint"),
            "reason": parsed.get("reason", "")
        }
        
