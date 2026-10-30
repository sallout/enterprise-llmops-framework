from pathlib import Path
import json,re
from pypdf import PdfReader
root=Path('output/dissertation')
metadata=json.loads((root/'document_metadata.json').read_text())
pdf=next(Path('tmp/dissertation_build/render').glob('*.pdf'))
r=PdfReader(pdf);texts=[p.extract_text() for p in r.pages]
normalize=lambda s:re.sub(r'\s+',' ',s).strip()
intro=next(i for i,t in enumerate(texts) if re.match(r'^1\s+1 Introduction\s+1\.1',t))
mp={};missing=[]
front=['Acknowledgements','Abstract sheet','Supervisor certificate','Contents','List of figures and tables','Abbreviations']
for heading in front:
 for i,t in enumerate(texts[:intro]):
  if heading in [x.strip() for x in t.splitlines()]:
   mp[heading]=t.splitlines()[0].strip();break
for target in [h[0] for h in metadata['headings']]+metadata['captions']+['Final report checklist']:
 matched=False
 for i,t in enumerate(texts[intro:],intro):
  lines=[x.strip() for x in t.splitlines()]
  compact=normalize(t)
  if target in lines or normalize(target) in compact:
   mp[target]=str(i-intro+1);matched=True;break
 if not matched:missing.append(target)
source=(root/'final_dissertation.md').read_text()
terms=re.findall(r'^\| ([^|]+) \|',source.split('# Glossary')[1],re.M)[2:]
for term in terms:
 for i,t in enumerate(texts[intro:],intro):
  if re.search(r'\b'+re.escape(term)+r'\b',normalize(t),flags=re.I):
   mp['TERM:'+term]=str(i-intro+1);break
old_path=Path('tmp/dissertation_build/page_map.json');old=json.loads(old_path.read_text()) if old_path.exists() else {}
old_path.write_text(json.dumps(mp,indent=2))
print({'pages':len(r.pages),'intro_physical_page':intro+1,'mapped':len(mp),'missing':missing,'changed':sum(old.get(k)!=v for k,v in mp.items()),'pdf_bytes':pdf.stat().st_size})
for i,t in enumerate(texts):print(i+1,repr(t[:70]))
