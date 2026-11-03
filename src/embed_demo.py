"""הדגמה: איך נראה embedding ואיך מודדים קרבה בין טקסטים.

הרצה מתיקיית הפרויקט:  python src/embed_demo.py
בהרצה הראשונה המודל יורד מהאינטרנט (כמה מאות MB), וזה לוקח זמן.
"""
import math

from llama_index.core import Settings

from config import setup_embeddings

setup_embeddings()
model = Settings.embed_model


def cosine(a, b):
    """cosine similarity: ככל שקרוב ל-1, המשמעות דומה יותר."""
    dot = sum(x * y for x, y in zip(a, b))
    return dot / (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)))


# השאלה
question = "כמה נקודות צריך כדי לנצח בקטאן?"
q_vec = model.get_query_embedding(question)

# שלושה קטעים: אחד רלוונטי, אחד ממשחק אחר, אחד לא קשור
passages = [
    "מי שמגיע ל-10 נקודות בתורו מנצח מיד. יישוב שווה נקודה אחת.",
    "המטרה היא לתת מט למלך של היריב. מט קורה כשהמלך מאוים.",
    "לכל שחקן יש 15 חיילים. מזיזים אותם סביב הלוח לפי הקוביות.",
]

print(f"אורך הווקטור: {len(q_vec)}")
print(f"5 המספרים הראשונים של השאלה: {[round(x, 3) for x in q_vec[:5]]}\n")

print(f"שאלה: {question}\n")
for text in passages:
    p_vec = model.get_text_embedding(text)
    print(f"{cosine(q_vec, p_vec):.3f}  <-  {text}")
