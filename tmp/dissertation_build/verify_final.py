from pathlib import Path
from zipfile import ZipFile
from lxml import etree as E
from pypdf import PdfReader
import json, re, shutil, hashlib
root=Path('/Users/mas/Development/DissertationProject')
out=root/'output/dissertation'
pdf=out/'Bridging_the_Chasm_Final_Report.pdf'
shutil.copy2(root/'tmp/dissertation_build/render'/pdf.name,pdf)
docx=out/'Bridging_the_Chasm_Final_Report.docx'
r=PdfReader(pdf); texts=[p.extract_text() for p in r.pages]
assert len(texts)==47 and all(t.strip() for t in texts)
assert '1 Introduction' in texts[9] and texts[9].strip().startswith('1\n')
assert 'Final report checklist' in texts[-1]
assert pdf.stat().st_size < 10_000_000
ns={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
with ZipFile(docx) as z:
    xml=E.fromstring(z.read('word/document.xml'))
    comments=E.fromstring(z.read('word/comments.xml'))
    ids=comments.xpath('//w:comment/@w:id',namespaces=ns)
    anchors=xml.xpath('//w:commentRangeStart/@w:id',namespaces=ns)
    assert sorted(ids)==sorted(anchors)
    sizes=xml.xpath('//w:sectPr/w:pgSz',namespaces=ns)
    assert all(int(s.get('{'+ns['w']+'}w'))==12960 and int(s.get('{'+ns['w']+'}h'))==15840 for s in sizes)
    assert len(xml.xpath('//w:drawing',namespaces=ns))==2
source=(out/'final_dissertation.md').read_text()
body,refs=source.split('# References',1)
assert {int(n) for group in re.findall(r'\[([0-9, ]+)\]',body) for n in re.findall(r'\d+',group)}==set(range(1,18))
metadata=json.loads((out/'document_metadata.json').read_text())
assert metadata['abstract_word_count']<=200
assert 'codeplan' not in (' '.join(texts)).lower()
assert 'codeplan' not in E.tostring(xml).decode().lower()
result={'date':'2026-10-30','pdf_pages':47,'pdf_bytes':pdf.stat().st_size,'pdf_text_extractable_every_page':True,'introduction_physical_page':10,'introduction_logical_page':1,'last_page':'Final report checklist','abstract_words':metadata['abstract_word_count'],'references_cited':17,'figures':2,'tables':11,'word_completion_comments':len(ids),'visual_review':{'pages_reviewed':list(range(1,48)),'method':'Rendered page PNGs inspected visually; no clipping, overlap or broken tables found.'},'tests_rerun_for_document_task':False,'training_status':'Larger run completion and independent model evaluation pending; operator-supplied server evidence distinguished from local artifacts.','files':{p.name:{'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size} for p in (docx,pdf)}}
(out/'document_verification.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
