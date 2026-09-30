"""记录/核对人工复核过的模型卡与引用源内容；文件mtime不能证明文档新旧。

复核MODEL_CARD及变化的引用源之后：python -m src.eval.modelcard_review --record-review
默认只核对，不更新任何记录。这不是对语义正确性的自动背书。
"""
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
RECORD=ROOT/'docs'/'audits'/'MODEL_CARD_SOURCES.json'


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def snapshot():
    card=ROOT/'MODEL_CARD.md';sources={}
    for name in sorted(set(re.findall(r'[a-zA-Z0-9_/-]+\.(?:md|json)',card.read_text()))):
        candidates=[ROOT/name,ROOT/'reports'/name]
        p=next((p for p in candidates if p.is_file() and p!=card and p!=RECORD),None)
        if p is not None: sources[p.relative_to(ROOT).as_posix()]=digest(p)
    return {'model_card_sha256':digest(card),'sources':sources,
            'scope':'人工复核后的内容快照；校验变化，不证明语义或真实上线有效性'}


def main():
    current=snapshot()
    if '--record-review' in sys.argv:
        RECORD.write_text(json.dumps(current,ensure_ascii=False,indent=2)+'\n')
        print(f"记录已复核引用源：{len(current['sources'])}份")
    elif not RECORD.exists() or current!=json.loads(RECORD.read_text()):
        raise SystemExit('模型卡或引用源发生变化；需人工复核后重新记录')
    else: print(f"模型卡及{len(current['sources'])}份引用源内容一致")


if __name__=='__main__':main()
