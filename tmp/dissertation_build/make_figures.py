from pathlib import Path
from PIL import Image,ImageDraw,ImageFont

OUT=Path('output/dissertation/figures')
FONT='/System/Library/Fonts/Supplemental/Arial.ttf'
BOLD='/System/Library/Fonts/Supplemental/Arial Bold.ttf'


def font(size,bold=False):return ImageFont.truetype(BOLD if bold else FONT,size)


def label(d,box,text,size=33,bold=False):
    x0,y0,x1,y1=box
    lines=text.split('\n');f=font(size,bold);height=len(lines)*(size+9)
    for i,line in enumerate(lines):
        bounds=d.textbbox((0,0),line,font=f);width=bounds[2]-bounds[0]
        d.text(((x0+x1-width)/2,(y0+y1-height)/2+i*(size+9)),line,font=f,fill='black')


def rect(d,box,text,fill='#F4F4F4',size=33):
    d.rounded_rectangle(box,radius=10,outline='black',width=3,fill=fill);label(d,box,text,size)


def arrow(d,start,end):
    import math
    d.line([start,end],fill='black',width=4)
    angle=math.atan2(end[1]-start[1],end[0]-start[0]);x,y=end
    pts=[end,(x-18*math.cos(angle-.45),y-18*math.sin(angle-.45)),(x-18*math.cos(angle+.45),y-18*math.sin(angle+.45))]
    d.polygon(pts,fill='black')


im=Image.new('RGB',(1800,1250),'white');d=ImageDraw.Draw(im)
rect(d,(40,35,830,160),'Policy PDFs and repository source\nOptional existing accepted exports')
rect(d,(970,35,1760,160),'Native NVIDIA datasets\nMetadata and revision inspection')
arrow(d,(435,160),(435,225));arrow(d,(1365,160),(1365,225))
rect(d,(40,225,830,375),'Parse and generate candidates\nSource checks and inherited test evidence')
rect(d,(970,225,1760,375),'Stream, adapt and filter\nApproved native selection and hashes')
arrow(d,(435,375),(435,440));arrow(d,(1365,375),(1365,440))
rect(d,(40,440,830,565),'Enterprise train and holdout splits\nPrivate publication and restoration')
rect(d,(970,440,1760,565),'Source plan and decision database\nExact approved-row replay')
arrow(d,(435,565),(435,625));arrow(d,(1365,565),(1365,625))
rect(d,(235,625,1565,755),'Fresh global fingerprint gate and exact Qwen tokenization\nAll holdouts indexed first; only train rows cached',size=35)
arrow(d,(900,755),(900,810))
rect(d,(235,810,1565,935),'Training-only calibration and physical MLP channel pruning\nBF16 LoRA recovery and adapter merge',size=35)
arrow(d,(900,935),(900,990))
rect(d,(235,990,1565,1100),'NF4 4-bit save and reload\nTechnical export evidence',size=35)
arrow(d,(900,1100),(900,1150));label(d,(100,1150,1700,1235),'Independent quality, efficiency and robustness evaluation: pending',size=32)
im.save(OUT/'workflow.png',optimize=True)

im=Image.new('RGB',(1800,730),'white');d=ImageDraw.Draw(im)
label(d,(0,0,1800,70),'Physical channel removal preserves matching projection dimensions',size=38,bold=True)
rect(d,(30,240,230,440),'Hidden\ninput',size=33)
rect(d,(320,110,650,270),'Gate projection\nSiLU',size=33)
rect(d,(320,395,650,555),'Up projection',size=33)
arrow(d,(230,290),(320,190));arrow(d,(230,390),(320,475))
d.ellipse((755,265,910,420),outline='black',width=4);label(d,(755,265,910,420),'Multiply',size=28)
arrow(d,(650,190),(775,280));arrow(d,(650,475),(775,405))
for i in range(6):
    y=150+i*70
    fill='#D5D5D5' if i==2 else '#FFFFFF'
    d.rectangle((1040,y,1160,y+40),fill=fill,outline='black',width=3)
    if i==2:
        d.line((1030,y-7,1170,y+47),fill='black',width=4)
        label(d,(1175,y-8,1400,y+48),'Remove j',size=27)
arrow(d,(910,342),(1040,342))
rect(d,(1430,215,1760,435),'Down projection\nHidden output',size=33)
arrow(d,(1160,220),(1430,290));arrow(d,(1160,500),(1430,370))
label(d,(50,585,1750,710),'For channel j: delete gate and up rows, and the matching down column.\nAttention, hidden width, layer count and vocabulary remain unchanged.',size=32)
im.save(OUT/'pruning.png',optimize=True)

