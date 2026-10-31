"""Append a transparent two-page follow-up to the six-page submitted baseline report."""
from pathlib import Path
import json,io
from reportlab.platypus import SimpleDocTemplate,Paragraph,Table,TableStyle,Spacer,PageBreak
from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from pypdf import PdfReader,PdfWriter

root=Path(__file__).resolve().parent; run=root/'code/runs'; reports=root/'reports'
exp=json.loads((run/'v2-search/experiments-v2.json').read_text()); cv=json.loads((run/'v2-search/cross-validation.json').read_text()); stats=json.loads((reports/'analysis-v2.json').read_text()); frozen=json.loads((run/'freeze-v2.json').read_text()); test=json.loads((run/'v2-test-score.json').read_text()); res=json.loads((run/'v2-test-resources.json').read_text())
styles=getSampleStyleSheet(); styles.add(ParagraphStyle(name='BodyV2',fontName='Helvetica',fontSize=10,leading=14,spaceAfter=9,textColor=colors.HexColor('#243345'))); styles.add(ParagraphStyle(name='HeadV2',fontName='Helvetica-Bold',fontSize=16,leading=20,spaceAfter=13,textColor=colors.HexColor('#123149'))); styles.add(ParagraphStyle(name='SubV2',fontName='Helvetica-Bold',fontSize=11,leading=15,spaceBefore=5,spaceAfter=6,textColor=colors.HexColor('#087E8B'))); styles.add(ParagraphStyle(name='SmallV2',parent=styles['BodyV2'],fontSize=8.5,leading=11))
story=[]; md=[]
def p(t,style='BodyV2'): story.append(Paragraph(t,styles[style])); md.append(t)
def table(headers,rows,widths):
    data=[[Paragraph(str(x),styles['SmallV2']) for x in row] for row in [headers]+rows]; t=Table(data,colWidths=widths,repeatRows=1)
    t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#E7F1F4')),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#F5F7FA')]),('VALIGN',(0,0),(-1,-1),'TOP'),('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5),('LEFTPADDING',(0,0),(-1,-1),7)])); story.extend([t,Spacer(1,9)]); md.append(str([headers]+rows))

p('6. Follow-up: optimizing the ranking objective','HeadV2')
p('The original submitted verifier and result on pages 1-6 remain a separately frozen artifact. This follow-up tests a new hypothesis: directly optimizing within-question choice, rather than independent candidate correctness, may better match selection accuracy. It is reported in full even though its final test result does not improve.')
p('Protocol and training objective','SubV2')
p('The follow-up uses only supplied training inputs/references and validation development. The original test score was already public; no test inputs or references enter the follow-up training, search or three-fold diagnostic. The new method is selected on validation and frozen before its one final test evaluation. No parameters are changed afterward.')
p('Input features are the 1,024 frozen embedding dimensions plus the original 46 statistics (1,070 total); the centered duplicate block is omitted. Scaling is fitted on training candidates only. Training focuses on the 2,114 training groups with at least one valid wrong candidate. Groups with no valid wrong answer need no learned discrimination after invalid-answer masking. No IDs or stored choices are features.')
p('The selected listwise loss is the mean of logsumexp over valid candidate logits minus logsumexp over correct candidate logits, plus 0.1 times squared weight norm divided by two. This rewards probability assigned to any correct candidate. A linear model is optimized by L-BFGS-B. The selected fit converges in 28 iterations; weights are exported as float32 for NumPy-only inference.')
p('For candidates sharing a parsed answer, pooled score is T * log(mean(exp(z/T))), with T = 0.5. The highest-scoring answer group wins. Log-mean pooling avoids automatically multiplying evidence by the number of duplicate answers. Invalid parsed candidates remain masked.')
table(['Validation comparison','Accuracy %'],[['Original submitted centered BCE LR','92.23'],['BCE LR with combined features','92.11'],['Best pairwise combined LR','92.73'],['Best uniform-positive listwise combined model','93.48'],['Selected marginal listwise model, no pooling','93.86'],['Selected marginal model with answer pooling','94.24']],[382,113])
p('The bounded search fits 20 models: eight pairwise logistic models and twelve linear listwise models. It then compares ten fixed pooling settings per model plus the original baseline, for 210 validation comparisons. The winner solves 752/798 validation questions, 16 more than the original 736/798. The no-pooling comparison isolates the final pooling step; other comparisons change the objective and training-group treatment together.','SmallV2')

story.append(PageBreak())
p('7. Follow-up evidence, costs and negative result','HeadV2')
table(['Evaluation','Original verifier','Follow-up verifier'],[['Validation','736/798 (92.23%)','752/798 (94.24%)'],['Informative validation subset','293/355 (82.54%)','309/355 (87.04%)'],['Training-only 3-fold held-out groups','4318/4755 (90.81%)','4329/4755 (91.04%)'],['Final official test','380/467 (81.37%)','374/467 (80.09%)']],[210,142,143])
p('<b>The final test improvement did not materialize.</b> The follow-up loses six test questions (-1.28 percentage points), despite the validation gain. The already-submitted baseline is left unchanged. The follow-up weights, freeze record, predictions and independent scorer JSON are retained as a transparent failed experiment, not presented as an improved leaderboard result.')
p('Reliability and interpretation','SubV2')
p('On validation, the follow-up fixes 29 original mistakes and introduces 13. The paired 5,000-resample question bootstrap gives a +0.50 to +3.63 percentage-point gain interval. This interval is descriptive and does not correct for selecting among 210 validation comparisons. The much smaller training-only three-fold gain (+0.23 points) provides a caution against treating the +2.01-point validation gain as robust. The three-fold configuration is itself chosen after validation search, so this is a robustness diagnostic, not an untouched confirmatory study.')
p('Possible explanations include development-set selection effects, split difficulty and sampling variation. The available results do not identify a single cause. No test-error-driven feature changes or second test-based search were performed. The finding supports caution about replacing a simple verifier solely because a more task-aligned loss looks better on one validation bank.')
p('Resources, reproduction and checks','SubV2')
p(f"CPU only: Intel Core i7-13650HX, no GPU. Follow-up development takes {exp['total_seconds']:.2f} s; the three-fold diagnostic adds {cv['seconds']:.2f} s. The frozen inference bundle contains {frozen['inference_asset_bytes']:,} bytes, versus the original 26,397 bytes. Complete fresh-process test prediction plus scoring takes {res['total_wall_seconds']:.3f} s locally with four threads. These are local measurements; the reference Xeon machine was not tested.")
p('From code/, install requirements-training.txt and run train_v2.py to reproduce the search, or cross_validate_v2.py for the training-only diagnostic. For checkpoint reproduction, install requirements.txt and run evaluate_v2.py --split validation or --split test. The unchanged official runner/scorer are used. Inference assets are in runs/model_bundle_v2; runs/freeze-v2.json records matching source/asset hashes before final scoring. The finite-difference gradient error is 2.01e-8; float32 export preserves every validation choice; renamed IDs, a seven-question bank and candidate permutation checks pass.')
p('AI assistance and source attribution on page 6 also apply to this follow-up. Codex provided substantial design, implementation, execution, analysis and writing assistance. This follow-up adds no external data or pretrained weights and uses the same pinned dependencies. Frozen source and weights remain unchanged after the reported test.','SmallV2')

def chrome(c,d):
    c.saveState(); w,h=A4; c.setFont('Helvetica',8); c.setFillColor(colors.HexColor('#536579')); c.drawString(50,h-27,'DASE7506 / FOLLOW-UP RANKING EXPERIMENT'); c.setStrokeColor(colors.HexColor('#D4E0E6')); c.line(50,h-35,w-50,h-35); c.line(50,36,w-50,36); c.drawString(50,24,'Negative result reported without test-based retuning'); c.drawRightString(w-50,24,str(d.page+6)); c.restoreState()
buffer=io.BytesIO(); SimpleDocTemplate(buffer,pagesize=A4,leftMargin=50,rightMargin=50,topMargin=53,bottomMargin=51).build(story,onFirstPage=chrome,onLaterPages=chrome)
appendix=PdfReader(io.BytesIO(buffer.getvalue())); assert len(appendix.pages)==2
writer=PdfWriter(); baseline=PdfReader(reports/'report.pdf'); assert len(baseline.pages)==6
for page in baseline.pages: writer.add_page(page)
for page in appendix.pages: writer.add_page(page)
writer.add_metadata({'/Title':'Selecting Reliable LLM Reasoning - Full Report with Follow-up','/Author':'DASE7506 MP2 coursework'})
with (reports/'optimization-report.pdf').open('wb') as f: writer.write(f)
(reports/'optimization-notes.md').write_text('\n\n'.join(md)+'\n',encoding='utf-8')
print('Created 8-page report:',reports/'optimization-report.pdf')
