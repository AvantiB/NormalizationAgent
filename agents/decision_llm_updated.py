import json, requests, re
from typing import Any, Dict, List, Optional


class DecisionAgentLLM:
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

    # --------------------------------------------------
    # NEW: Verifier (single candidate accept/reject)
    # --------------------------------------------------
    def verify(self, entity: Dict[str, Any], candidate: Dict[str, Any]) -> Dict[str, Any]:
        """
        Verify whether ONE candidate is a valid match.
        Output schema:
        {
          "accept": bool,
          "confidence": float 0..1,
          "notes": str
        }
        """
        system_prompt = (
            "You are a clinical terminology NORMALIZATION VERIFIER.\n"
            "Task: Decide whether the provided candidate concept matches the entity mention in context.\n"
            "Rules:\n"
            "- You are NOT selecting among many options. You ONLY verify this ONE candidate.\n"
            "- Accept ONLY if the candidate meaning matches the entity in the given context.\n"
            "- If the context indicates negation/absence, STILL verify the concept meaning (not presence).\n"
            "- If the candidate adds specificity not supported by text/context, reject.\n"
            "- Output STRICT JSON only.\n"
        )

        user_prompt = (
            f"- Entity text (verbatim): \"{entity['verbatim']}\"\n"
            f"- Entity text (abbreviation expanded or normalized): \"{entity['expanded']}\"\n"
            f"- Clinical note context: \"{entity['context']}\"\n\n"
            f"- Semantic Tag hint: \"{entity['semantic_tag_hint']}\"\n\n"
            "Candidate concept (one):\n"
            + json.dumps(candidate, ensure_ascii=False, indent=2)
            + "\n\n"
            "Return STRICTLY ONLY valid JSON and nothing else:\n"
            "{\n"
            "  \"accept\": true or false,\n"
            "  \"confidence\": float between 0 and 1,\n"
            "  \"notes\": \"short reason\"\n"
            "}"
        )

        content = self._call_llm(system_prompt, user_prompt)
        if not content:
            return {"accept": False, "confidence": 0.0, "notes": "llm_error", "source": "llm_verify_error"}

        parsed = self._safe_json_loads(content)
        if not parsed or "accept" not in parsed:
            print("LLM JSON parse failure (verify)")
            print("RAW OUTPUT:", content)
            return {"accept": False, "confidence": 0.0, "notes": "parse_failure", "source": "llm_verify_parse_failure"}

        parsed["source"] = "llm_verify"
        # normalize types defensively
        parsed["accept"] = bool(parsed.get("accept", False))
        try:
            parsed["confidence"] = float(parsed.get("confidence", 0.0))
        except Exception:
            parsed["confidence"] = 0.0
        parsed["notes"] = str(parsed.get("notes", "")).strip()
        return parsed

    # --------------------------------------------------
    # NEW: Selector (multi-candidate choose-one-or-null)
    # --------------------------------------------------
    def select(self, entity: Dict[str, Any], candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Select the best concept among multiple candidates OR return null.
        Output schema (same as your old decide()):
        {
          "concept_id": str|null,
          "concept_name": str|null,
          "hierarchy": str|null,
          "confidence": float 0..1
        }
        """
        if not candidates:
            return {
                "concept_id": None,
                "concept_name": None,
                "hierarchy": None,
                "confidence": 0.0,
                "source": "no_candidates"
            }

        allowed_ids = {c["concept_id"] for c in candidates if c.get("concept_id") is not None}

        system_prompt = (
            "You are a careful clinical terminology normalization assistant.\n"
            "You must select the single best matching concept from the provided candidates by considering the entity within the provided clinical note context.\n"
            "Rules:\n"
            "- Use ONLY the provided candidates.\n"
            "- Do NOT infer details not present in the entity/context.\n"
            "- Prefer the least-specific concept that matches, unless the text explicitly includes the specificity.\n"
            "- Use hierarchy/semantic tag appropriately.\n"
            "- If none are acceptable, return null.\n"
            "Output STRICT JSON only.\n"
        )

        user_prompt = (
            f"- Entity text (verbatim): \"{entity['verbatim']}\"\n"
            # f"- Entity text (abbreviation expanded): \"{entity['expanded']}\"\n"
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

        content = self._call_llm(system_prompt, user_prompt)
        if not content:
            return {
                "concept_id": None,
                "concept_name": None,
                "hierarchy": None,
                "confidence": 0.0,
                "source": "llm_error"
            }

        parsed = self._safe_json_loads(content)
        if not parsed:
            print("LLM JSON parse failure (select)")
            print("RAW OUTPUT:", content)
            return {
                "concept_id": None,
                "concept_name": None,
                "hierarchy": None,
                "confidence": 0.0,
                "source": "parse_failure",
            }

        cid = parsed.get("concept_id")
        if cid is not None and cid not in allowed_ids:
            print("LLM hallucinated concept_id:", cid)
            return {
                "concept_id": None,
                "concept_name": None,
                "hierarchy": None,
                "confidence": 0.0,
                "source": "llm_hallucination",
            }

        parsed["source"] = "llm_selector"
        return parsed

    # --------------------------------------------------
    # Backward compatibility (if other code calls decide)
    # # --------------------------------------------------
    # def decide(self, entity: Dict[str, Any], candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
    #     return self.select(entity, candidates)
