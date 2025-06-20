import re
import inflect
p = inflect.engine()

PLURAL_MAP = {"courses": "course", "subjects": "subject"}
VERB_CONJUGATIONS = {"does": "do", "has": "have"}
ABBREVIATIONS = {"u": "you", "r": "are"}
SYNONYMS = {"teachers": "academic staff"}

DEPARTMENT_ALIASES = {"computer science": ["csc", "cs"], "law": ["law dept", "faculty of law"]}

def normalize_text(text):
    text = text.lower().strip()
    text = re.sub(r'([a-z]{2,4})[\s-]?(\d{3})', r'\1\2', text)
    words = []
    for word in text.split():
        if word in VERB_CONJUGATIONS:
            words.append(VERB_CONJUGATIONS[word])
        elif word in PLURAL_MAP:
            words.append(PLURAL_MAP[word])
        else:
            singular = p.singular_noun(word)
            words.append(singular if singular else word)
    text = " ".join(words)
    for k, v in {**ABBREVIATIONS, **SYNONYMS}.items():
        text = re.sub(rf"\b{k}\b", v, text)
    for dept, aliases in DEPARTMENT_ALIASES.items():
        for alias in aliases:
            text = re.sub(rf"\b{alias}\b", dept, text)
    return text

def extract_course_codes(text):
    return [match.replace(" ", "").replace("-", "") for match in re.findall(r'\b([a-z]{2,4}\s?-?\s?\d{3})\b', text.lower())]
