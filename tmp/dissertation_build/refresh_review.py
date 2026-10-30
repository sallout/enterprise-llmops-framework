from pathlib import Path
from PIL import Image,ImageDraw
from pypdf import PdfReader
root=Path('tmp/dissertation_build/render');count=len(PdfReader(next(root.glob('*.pdf'))).pages)
out=Path('tmp/dissertation_build/revised_pairs');out.mkdir(exist_ok=True)
for start in range(1,count+1,2):
 images=[Image.open(root/f'page-{n}.png').convert('RGB') for n in range(start,min(start+2,count+1))]
 canvas=Image.new('RGB',(sum(im.width for im in images)+18, max(im.height for im in images)+30),'#dddddd');x=0
 for n,im in enumerate(images,start):
  canvas.paste(im,(x,30));ImageDraw.Draw(canvas).text((x+10,8),f'Physical page {n}',fill='black');x+=im.width+18
 canvas.save(out/f'pages-{start:02d}.png')
print(count)
p=Path('tmp/dissertation_build/make_figures.py');s=p.read_text();s=s[:s.index("im=Image.new('RGB',(1800,1160)")];p.write_text(s)
Path('output/dissertation/figures/codeplan_boundary.png').unlink(missing_ok=True)
