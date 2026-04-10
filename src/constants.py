"""
Static mappings shared across the pipeline.
"""

NDA_TO_H = {
    "nda-1":  "H01",
    "nda-2":  "H02",
    "nda-3":  "H03",
    "nda-4":  "H04",
    "nda-5":  "H05",
    "nda-7":  "H06",
    "nda-8":  "H07",
    "nda-10": "H08",
    "nda-11": "H09",
    "nda-12": "H10",
    "nda-13": "H11",
    "nda-15": "H12",
    "nda-16": "H13",
    "nda-17": "H14",
    "nda-18": "H15",
    "nda-19": "H16",
    "nda-20": "H17",
}

H_TO_NDA = {v: k for k, v in NDA_TO_H.items()}

LABEL_MAP = {
    "Entailment":   "ENTAILED",
    "Contradiction": "CONTRADICTED",
    "NotMentioned": "NOT_MENTIONED",
}

LABEL_TO_STATUS = {
    "ENTAILED":      "satisfied",
    "CONTRADICTED":  "conflict",
    "NOT_MENTIONED": "missing",
}

HYPOTHESES = {
    "H01": "All Confidential Information shall be expressly identified by the Disclosing Party.",
    "H02": "Confidential Information shall only include technical information.",
    "H03": "Confidential Information may include verbally conveyed information.",
    "H04": "Receiving Party shall not use any Confidential Information for any purpose other than the purposes stated in Agreement.",
    "H05": "Receiving Party may share some Confidential Information with some of Receiving Party's employees.",
    "H06": "Receiving Party may share some Confidential Information with some third-parties (including consultants, agents and professional advisors).",
    "H07": "Receiving Party shall notify Disclosing Party in case Receiving Party is required by law, regulation or judicial process to disclose any Confidential Information.",
    "H08": "Receiving Party shall not disclose the fact that Agreement was agreed or negotiated.",
    "H09": "Receiving Party shall not reverse engineer any objects which embody Disclosing Party's Confidential Information.",
    "H10": "Receiving Party may independently develop information similar to Confidential Information.",
    "H11": "Receiving Party may acquire information similar to Confidential Information from a third party.",
    "H12": "Agreement shall not grant Receiving Party any right to Confidential Information.",
    "H13": "Receiving Party shall destroy or return some Confidential Information upon the termination of Agreement.",
    "H14": "Receiving Party may create a copy of some Confidential Information in some circumstances.",
    "H15": "Receiving Party shall not solicit some of Disclosing Party's representatives.",
    "H16": "Some obligations of Agreement may survive termination of Agreement.",
    "H17": "Receiving Party may retain some Confidential Information even after the return or destruction of Confidential Information.",
}

SYSTEM_PROMPT = (
    "You are a contract NLI system. Given a list of numbered contract spans and "
    "a set of hypotheses, classify each hypothesis as one of:\n"
    "- ENTAILED: the contract entails the hypothesis\n"
    "- CONTRADICTED: the contract contradicts the hypothesis\n"
    "- NOT_MENTIONED: the contract does not mention this\n\n"
    "For ENTAILED and CONTRADICTED, you must identify the span IDs that serve as "
    "evidence. Pay careful attention to exceptions introduced by phrases like "
    "\"notwithstanding the foregoing\", \"except\", \"provided however\" — these can "
    "flip the meaning of an earlier general rule.\n\n"
    "Return ONLY a JSON array. No other text, no markdown fences."
)
