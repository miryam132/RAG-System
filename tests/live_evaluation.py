"""Run real Gemini evaluation locally: python tests/live_evaluation.py.
Uses paid APIs, builds a temporary c512 index, saves actual answers and sources.
Review answers manually; expected text is a rubric, not a generated result.
"""
import truststore
truststore.inject_into_ssl()

import json
import sys
import tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import config
import ingest
import rag

CASES = [
    ('כמה קלפים מקבל כל שחקן באונו בתחילת המשחק?', '7 קלפים', 'אונו', 'הכנה'),
    ('כמה משאבים צריך כדי לבנות עיר בקטאן?', 'שלוש פלדה ושתי חיטה', 'קטאן', 'מחירי בנייה'),
    ('באילו תנאים מותר לבצע הצרחה בשחמט?', 'המלך והצריח לא זזו, הדרך פנויה, המלך אינו בשח ואינו עובר או נוחת במשבצת מאוימת', 'שחמט', 'צריחה'),
    ('האם 12, 13, 1 הם רצף חוקי ברמיקוב?', 'לא', 'רמיקוב', 'סוגי סדרות'),
    ('מי המציא את משחק קטאן?', rag.NO_ANSWER, None, None),
    ('איך משחקים בקטאן?', 'תשובה המבוססת על מקורות קטאן', 'קטאן', None),
    ('וכמה נקודות צריך כדי לנצח?', '10 נקודות בתורך; שאלת המשך לקטאן', 'קטאן', 'ניצחון'),
]

def main():
    try:
        config.setup_embeddings()
        config.setup_llm()
    except RuntimeError as error:
        print(f'BLOCKED: {error}')
        return 2
    output = Path(__file__).with_name('live-results.json')
    results = []
    with tempfile.TemporaryDirectory(prefix='rag-live-eval-') as directory:
        config.CHROMA_DIR = Path(directory)
        ingest.build_index(ingest.load_sections(), 512, 50, config.COLLECTION_NAME + '_c512')
        index = rag.load_index('c512')
        history = []
        for i, (question, expected, game, section) in enumerate(CASES):
            record = {'question':question, 'expected_manual_rubric':expected,
                      'expected_game':game, 'expected_section':section}
            try:
                result = rag.ask(index, question, history=history if i == 6 else None)
                record.update(result)
                if game:
                    record['expected_source_retrieved'] = any(s['game'] == game and (section is None or s['section'] == section) for s in result['sources'])
                if i == 4:
                    record['abstained_exactly'] = result['answer'].strip() == rag.NO_ANSWER
                if i == 5:
                    history = [{'role':'user','content':question}, {'role':'assistant','content':result['answer']}]
            except Exception as error:
                record['error_type'] = type(error).__name__
            results.append(record)
            output.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Actual outputs saved to {output}; answer correctness requires manual review.')
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
