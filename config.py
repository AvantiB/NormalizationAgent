STRICT_AUTO_THRESHOLD = 0.90
LLM_THRESHOLD = 0.60

STOPWORDS = {
    "of","in","the","a","an","and","or","to","for","with","on",
    "at","by","from","without","due","as","is","was"
}

TAU_HIGH = 0.85   # if SapBERT@1 >= this -> verifier stage
TAU_LOW  = 0.65   # if SapBERT@1 in [TAU_LOW, TAU_HIGH) -> selector stage

FUSION_TOP_K_FOR_LLM = 16