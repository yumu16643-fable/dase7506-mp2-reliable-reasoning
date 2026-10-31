"""Append the GPU round to the preceding eight-page report (ten pages total)."""
from pathlib import Path
import io,json
from reportlab.platypus import SimpleDocTemplate,Paragraph,Table,TableStyle,Spacer,PageBreak
from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from pypdf import PdfReader,PdfWriter

root=Path(__file__).resolve().parent; runs=root/'code/runs'; reports=root/'reports'
e=json.loads((runs/'gpu-search/experiments-gpu.json').read_text()); a=json.loads((reports/'analysis-gpu.json').read_text()); resources=json.loads((runs/'gpu-validation-resources.json').read_text()); config=json.loads((runs/'model_bundle_gpu/config.json').read_text())
styles=getSampleStyleSheet(); styles.add(ParagraphStyle(name='GBody',fontName='Helvetica',fontSize=10,leading=14,spaceAfter=9,textColor=colors.HexColor('#243345'))); styles.add(ParagraphStyle(name='GHead',fontName='Helvetica-Bold',fontSize=16,leading=20,spaceAfter=13,textColor=colors.HexColor('#123149'))); styles.add(ParagraphStyle(name='GSub',fontName='Helvetica-Bold',fontSize=11,leading=15,spaceBefore=5,spaceAfter=6,textColor=colors.HexColor('#087E8B'))); styles.add(ParagraphStyle(name='GSmall',parent=styles['GBody'],fontSize=8.5,leading=11))
story=[]; md=[]
def p(t,style='GBody'): story.append(Paragraph(t,styles[style])); md.append(t)
def table(headers,rows,widths):
    data=[[Paragraph(str(x),styles['GSmall']) for x in r] for r in [headers]+rows]; t=Table(data,colWidths=widths,repeatRows=1)
    t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#E7F1F4')),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#F5F7FA')]),('VALIGN',(0,0),(-1,-1),'TOP'),('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5),('LEFTPADDING',(0,0),(-1,-1),7)])); story.extend([t,Spacer(1,9)]); md.append(str([headers]+rows))

p('8. GPU training with question-level model selection','GHead')
p('A third round investigates whether small nonlinear candidate-comparison networks can improve the compact verifier. Training runs on one NVIDIA GeForce RTX 4060 Laptop GPU (8 GB), below the permitted RTX 4090 limit. No additional pretrained weights or datasets are introduced. The original official submission and the CPU follow-up remain distinct frozen artifacts.')
p('Fixed scope and data discipline','GSub')
p('The plan is recorded in code/gpu_plan.json before training. Four architectures each use seeds 2026 and 2027: a shared MLP with binary loss, a shared MLP with marginal listwise loss, a contextual listwise network, and a contextual residual listwise network. Eight single models and four same-architecture two-seed logit ensembles form twelve candidate configurations. Architecture and seed selection use three-fold question-disjoint out-of-fold (OOF) accuracy inside the supplied training split, not validation ranking.')
p('Within each outer training fold, a further 15% question holdout selects the training epoch using loss with patience 12 and at most 100 epochs. The network is then reinitialized and refitted on the entire outer training fold for that epoch count before evaluating the outer holdout. Scaling and the residual baseline are refitted separately within the appropriate training portion. No candidate siblings cross a question-level boundary. The chosen final model is refitted on all training questions using the median selected epoch count, then evaluated on validation once.')
p('Networks and optimization','GSub')
p('The input is the original 1,024-dimensional embedding plus 46 text/answer statistics, standardized using fitted training statistics and clipped to [-5, 5]. A 64-unit ReLU projection gives h. Context models concatenate h, the four-candidate mean, the mean for candidates sharing a parsed numeric answer, and h times the group mean. A 32-unit ReLU layer produces a scalar score. This second nonlinear layer permits contextual effects; candidate permutations preserve corresponding scores.')
p('The residual network adds its learned score to a training-fitted version of the original logistic verifier. Training uses AdamW, learning rate 0.001, weight decay 0.1, dropout 0.3 and batches of 128 questions. Marginal listwise loss places probability on any correct valid candidate; all-valid-correct groups are omitted from listwise fitting because they require no learned discrimination after masking. The selected residual model uses seed 2027 and 11 final epochs.')
rows=[]
for arch in e['plan']['architectures']:
    single=max([c for c in e['candidates'] if c['architecture']==arch and len(c['seeds'])==1],key=lambda c:c['oof_accuracy'])
    ensemble=next(c for c in e['candidates'] if c['architecture']==arch and len(c['seeds'])==2)
    rows.append([arch.replace('_',' '),str(single['seeds'][0]),f"{100*single['oof_accuracy']:.2f}",f"{100*ensemble['oof_accuracy']:.2f}"])
table(['Architecture','Best seed','Single OOF %','Ensemble OOF %'],rows,[239,66,95,95])
p('All 24 outer-fold model evaluations, their inner-selected epoch counts and costs are retained in experiments-gpu.json. Validation labels are opened only after the training-OOF winner is fixed. No public test bank is evaluated in this round.','GSmall')

story.append(PageBreak()); p('9. GPU results, qualification gate and reproduction','GHead')
table(['Measure','Original submitted model','GPU-trained experiment'],[['Training-only OOF accuracy','4318/4755 (90.81%)','4338/4755 (91.23%)'],['Validation accuracy','736/798 (92.23%)','739/798 (92.61%)'],['Public test accuracy','380/467 (81.37%)','Not evaluated this round'],['Required inference assets','26,397 bytes','344,253 bytes'],['Fresh CPU validation + scoring','1.275 s','2.091 s']],[207,144,144])
p('<b>There is a small development gain, but no demonstrated test gain.</b> Before training, the promotion gate required at least +0.5 percentage points in training-OOF accuracy and validation accuracy no lower than the original model. This is an experiment decision rule, not a course requirement. The measured OOF gain is +0.42 points, below the gate, so no new public test evaluation or leaderboard submission is made. The existing submission #340 remains unchanged.')
p('Evidence and limitations','GSub')
p('The selected network fixes 127 original OOF mistakes and introduces 107, a net gain of twenty questions. A descriptive paired bootstrap with 5,000 question resamples gives a gain interval of -0.21 to +1.05 percentage points. It includes zero and does not correct for selecting among twelve configurations or overlapping training folds. Validation fixes fifteen mistakes and introduces twelve, a net gain of only three. These results provide limited support for replacing the original compact verifier.')
p('This is not a controlled device-only accuracy comparison: the architecture, loss and model selection process also change. GPU execution enables the experiment, but using a GPU itself does not establish improved selection quality. The network needs about thirteen times the original inference assets and takes longer on the CPU validation run. Its extra complexity is not clearly justified by the small development gains.')
p('Resources, checks and exact commands','GSub')
p(f"Actual CUDA execution is verified on the RTX 4060. PyTorch is 2.6.0+cu118, CUDA runtime 11.8 and training Python 3.12.9. The full GPU development loop takes {e['total_seconds']:.2f} s after imports, including CPU preprocessing and baseline fitting. Peak PyTorch-allocated GPU memory is {e['peak_gpu_memory_bytes']/2**20:.2f} MiB. CPU prediction uses Python 3.12.7 and NumPy only; the two fresh official validation processes take {resources['total_wall_seconds']:.3f} s with four threads. The reference Xeon machine is not measured.")
p('CPU reproduction from code/: install requirements.txt, run download_data.py, then run python evaluate_gpu.py. The wrapper evaluates validation only and sets all thread limits. The frozen bundle is runs/model_bundle_gpu; runs/freeze-gpu.json records matching source and asset hashes. GPU retraining uses python -m pip install -r requirements-gpu.txt followed by python train_gpu.py. Exact package versions and device information are recorded in gpu-environment.json and gpu-hardware.txt.')
p('Automatic differentiation passes a finite-difference check. CUDA and NumPy inference select the same candidate on all 798 validation questions. Synthetic exports for every architecture, renamed IDs, a seven-question bank and candidate score permutation checks also pass. Neither source nor learned assets are changed after freezing. AI assistance and the data/software attribution on page 6 apply equally to this round.','GSmall')

def chrome(c,d):
    c.saveState(); w,h=A4; c.setFont('Helvetica',8); c.setFillColor(colors.HexColor('#536579')); c.drawString(50,h-27,'DASE7506 / GPU TRAINING EXPERIMENT'); c.setStrokeColor(colors.HexColor('#D4E0E6')); c.line(50,h-35,w-50,h-35); c.line(50,36,w-50,36); c.drawString(50,24,'GPU training / NumPy CPU inference / no public test search'); c.drawRightString(w-50,24,str(d.page+8)); c.restoreState()
buffer=io.BytesIO(); SimpleDocTemplate(buffer,pagesize=A4,leftMargin=50,rightMargin=50,topMargin=53,bottomMargin=51).build(story,onFirstPage=chrome,onLaterPages=chrome)
tail=PdfReader(io.BytesIO(buffer.getvalue())); assert len(tail.pages)==2
base=PdfReader(reports/'optimization-report.pdf'); assert len(base.pages)==8
writer=PdfWriter()
for page in base.pages: writer.add_page(page)
for page in tail.pages: writer.add_page(page)
writer.add_metadata({'/Title':'Selecting Reliable LLM Reasoning - CPU and GPU Experiments','/Author':'DASE7506 MP2 coursework'})
with (reports/'gpu-report.pdf').open('wb') as f: writer.write(f)
(reports/'gpu-notes.md').write_text('\n\n'.join(md)+'\n',encoding='utf-8'); print('Created ten-page report:',reports/'gpu-report.pdf')
