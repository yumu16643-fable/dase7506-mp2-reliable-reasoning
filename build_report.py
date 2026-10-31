"""Build the report from recorded experiments; no model fitting or test selection."""
from pathlib import Path
import json,sys
from xml.sax.saxutils import escape

ROOT=Path(__file__).resolve().parent
R=ROOT/'reports'; RUN=ROOT/'code/runs'
A=json.loads((R/'analysis.json').read_text())
E=json.loads((RUN/'search/experiments.json').read_text())
B=json.loads((R/'ablation_resources.json').read_text())
T=json.loads((RUN/'test-score.json').read_text())
F=json.loads((RUN/'freeze.json').read_text())
VRES=json.loads((RUN/'validation-resources.json').read_text())
TRES=json.loads((RUN/'test-resources.json').read_text())

def figures():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    out=R/'figures'; out.mkdir(exist_ok=True)
    labels=['Majority','Style only','Text statistics','Embeddings','Combined LR','Centered LR']
    vals=[86.2155,79.4486,89.7243,89.7243,92.1053,92.2306]
    fig,ax=plt.subplots(figsize=(8.2,2.8))
    bars=ax.barh(labels,vals,color=['#9BA8B8']*4+['#4C8EAE','#087E8B'])
    ax.set_xlim(70,96); ax.invert_yaxis(); ax.set_xlabel('Validation selection accuracy (%)'); ax.grid(axis='x',alpha=.15); ax.set_axisbelow(True)
    for bar,val in zip(bars,vals): ax.text(val+.25,bar.get_y()+bar.get_height()/2,f'{val:.2f}',va='center',fontsize=9)
    fig.tight_layout(); fig.savefig(out/'accuracy.png',dpi=180); plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(8.2,2.7))
    axes[0].plot([x['coverage']*100 for x in A['coverage']],[x['accuracy']*100 for x in A['coverage']],marker='o',color='#087E8B')
    axes[0].set(xlabel='Retained validation questions (%)',ylabel='Accuracy (%)',ylim=(85,101),title='Ranking by sigmoid confidence')
    axes[0].grid(alpha=.15)
    bins=A['confidence_bins']; axes[1].plot([0,1],[0,1],color='#9BA8B8',linestyle='--')
    axes[1].scatter([x['confidence'] for x in bins],[x['accuracy'] for x in bins],s=[18+x['n']/3 for x in bins],color='#087E8B',alpha=.8)
    axes[1].set(xlabel='Mean predicted confidence',ylabel='Observed accuracy',xlim=(0,1.03),ylim=(0,1.05),title='Equal-width bins; size reflects count')
    axes[1].grid(alpha=.15); fig.tight_layout(); fig.savefig(out/'reliability.png',dpi=180); plt.close(fig)

def build():
    from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,PageBreak,Image,Preformatted
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
    from reportlab.lib.enums import TA_LEFT
    from reportlab.lib.pagesizes import A4
    styles=getSampleStyleSheet()
    styles.add(ParagraphStyle(name='Body2',fontName='Helvetica',fontSize=10,leading=14,spaceAfter=8,textColor=colors.HexColor('#243345')))
    styles.add(ParagraphStyle(name='Small2',parent=styles['Body2'],fontSize=8.4,leading=11,spaceAfter=6))
    styles.add(ParagraphStyle(name='Title2',fontName='Helvetica-Bold',fontSize=25,leading=29,spaceAfter=17,textColor=colors.HexColor('#123149')))
    styles.add(ParagraphStyle(name='Section2',fontName='Helvetica-Bold',fontSize=16,leading=20,spaceAfter=12,textColor=colors.HexColor('#123149')))
    styles.add(ParagraphStyle(name='Sub2',fontName='Helvetica-Bold',fontSize=11,leading=15,spaceBefore=5,spaceAfter=6,textColor=colors.HexColor('#087E8B')))
    styles.add(ParagraphStyle(name='Code2',fontName='Courier',fontSize=8,leading=11,spaceAfter=9))
    story=[]; md=[]
    def p(text,style='Body2'):
        story.append(Paragraph(text,styles[style])); md.append(text.replace('<b>','**').replace('</b>','**').replace('<br/>','\n'))
    def h(text): p(text,'Sub2')
    def page(title):
        if story: story.append(PageBreak()); md.append('\n---\n')
        p(title,'Section2')
    def table(headers,rows,widths):
        data=[[Paragraph(escape(str(x)),styles['Small2']) for x in row] for row in [headers]+rows]
        t=Table(data,colWidths=widths,hAlign='LEFT',repeatRows=1)
        t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#E7F1F4')),('LINEBELOW',(0,0),(-1,0),.7,colors.HexColor('#A1C1CA')),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#F5F7FA')]),('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),7),('RIGHTPADDING',(0,0),(-1,-1),7),('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5)]))
        story.extend([t,Spacer(1,10)]); md.append('| '+' | '.join(headers)+' |'); md.append('| '+' | '.join(['---']*len(headers))+' |')
        md.extend('| '+' | '.join(map(str,row))+' |' for row in rows)
    def img(name,height):
        story.append(Image(str(R/'figures'/name),width=495,height=height)); story.append(Spacer(1,6)); md.append(f'![Experiment figure](figures/{name})')
    def code(text):
        story.append(Preformatted(text,styles['Code2'])); md.append('```\n'+text+'\n```')

    p('Selecting Reliable<br/>LLM Reasoning','Title2')
    p('DASE7506 - Mini-project 2 | 31 October 2026','Small2')
    p('A compact verifier combining frozen representations, answer agreement and trace statistics','Sub2')
    p('This project asks whether inexpensive candidate statistics complement supplied whole-trace representations, and whether nonlinear models justify their extra complexity. A standardized, regularized logistic verifier is selected using validation accuracy. It uses no external data, pretrained weights or inference-time generation.')
    table(['Frozen result','Evidence'],[
        ['Validation accuracy','92.23% (736 / 798); majority voting: 86.22%'],
        ['Official test accuracy',f"{T['accuracy']:.8f}% ({T['correct']} / {T['n']})"],
        ['Local test prediction + scoring',f"{TRES['total_wall_seconds']:.3f} s; four threads; fresh processes"],
        ['Required inference assets',f"{F['inference_asset_bytes']:,} bytes (0.025 MiB); limit: 16 MiB"]
    ],[175,320])
    h('Task and evaluation target')
    p('For each question, choose one of four supplied candidate solutions. A correct final numeric answer receives one point, even when several candidates are correct. This is answer selection, not certification that every intermediate reasoning step is valid. The official numeric parser and evaluator are unchanged.')
    table(['Split','Questions','Role'],[['Train','4,755','Fit features, scaling and model weights'],['Validation','798','Select configurations and analyze behavior'],['Test','467','Score the frozen method only']],[120,85,290])
    h('Main finding')
    p('Combining representations and simple statistics improves validation accuracy over either feature family alone. Increasing model complexity does not improve the selected score. However, performance drops on questions with multiple distinct answers; the overall validation score includes many easy consensus groups. The method is useful as a compact selector, but it does not reliably verify arithmetic or interpretation.')
    p('All recorded results are local measurements on the released course bank. The reference-machine time limit and performance on an instructor audit bank remain unverified.','Small2')

    page('1. Method and experimental protocol')
    h('Candidate features')
    table(['Feature family','Dimensions','Information used'],[
        ['Frozen embedding','1,024','Supplied whole-trace representation; no extra encoder'],
        ['Trace / answer statistics','46','Lengths; numeric and formatting counts; 19 lexical indicators; parsed answer properties; within-question agreement and relative statistics'],
        ['Centered embedding','1,024','Candidate embedding minus the mean of the four embeddings'],
        ['Final input','2,094','Embedding + centered embedding + 46 statistics']
    ],[135,70,290])
    p('The 46 statistics contain 11 surface / question-length features, six answer properties, 19 word-pattern counts and ten group-comparison features. They include answer validity, sign, integer status, log magnitude, answer frequency, distinct-answer count and relative length. No candidate index or question ID enters the model. Text statistics are not a semantic proof checker; the embedding carries the supplied representation of the reasoning.')
    h('Training and decision rule')
    p('Each candidate receives a binary label by comparing its parsed final answer with the training reference. All correct candidates are positive. StandardScaler is fit on the 19,020 training candidates only. A logistic model minimizes binary cross-entropy with L2 regularization; the selected inverse regularization strength is C = 0.01. Inference standardizes each feature vector and returns the largest logit among parseable candidates. If none parses, it returns the largest unmasked score. Ties use the first candidate.')
    code('x_ij = [embedding_ij, embedding_ij - mean_i, statistics_ij]\nz_ij = w^T ((x_ij - training_mean) / training_scale) + b\nchoice_i = argmax_j z_ij     (invalid parsed answers masked)')
    p('<b>Centering has an important limitation.</b> In a linear scorer, the group-mean contribution is identical across a question and cancels when comparing candidates. The extra centered block changes the fitted binary objective and effective regularization; it does not create nonlinear cross-candidate reasoning at inference. The agreement statistics do provide direct comparative information.')
    h('Development search')
    p('The fixed search evaluates two C values for agreement, style and text statistics, and three for embeddings, combined and centered features: 15 logistic fits. Six 64-unit ReLU MLPs use seeds 2026/2027; three equal-logit ensembles and two text-only histogram boosting models are also compared. MLPs use alpha = 1, Adam, batch size 256, at most 100 epochs and internal training-only early stopping. Boosting is an analysis-only comparison. External validation determines the submitted portable model; the test does not influence selection.')
    p('The MLP internal split is candidate-level and may contain siblings of training candidates. It is not an independent question-level performance estimate; all reported model selection results use the separate question-disjoint validation split.','Small2')

    page('2. Accuracy, ablations and computational cost')
    mapping=[('lr_agreement_0.1','Agreement LR'),('lr_style_0.1','Style LR'),('lr_text_0.1','Statistics LR'),('lr_embedding_0.1','Embedding LR'),('lr_combined_1.0','Combined LR'),('lr_context_0.01','Centered LR (selected)'),('mlp_combined_2026','Combined MLP, seed 2026'),('mlp_context_2027','Centered MLP, seed 2027'),('ensemble_combined','Combined MLP ensemble')]
    lookup={x['name']:x for x in B}; exp={x['name']:x for x in E['experiments']}
    rows=[]
    for name,label in mapping:
        v=lookup[name]; seconds=exp[name]['training_seconds']
        if name=='ensemble_combined': seconds=sum(exp[f'mlp_combined_{s}']['training_seconds'] for s in [2026,2027])
        rows.append([label,f"{v['accuracy']:.2f}",f'{seconds:.2f}',f"{v['total_wall_seconds']:.2f}",f"{v['asset_bytes']/1024:.1f}"])
    table(['Validation method','Acc. %','Fit s','Eval. s','Asset KiB'],rows,[213,61,61,70,90])
    p('Eval. s includes two fresh official processes: prediction and scoring, startup, imports, data hashing/loading and preprocessing. Assets include each measured configuration; the submitted configuration adds descriptive metadata. Measurements are single local runs, with the OS disk cache unflushed. Fit s excludes shared feature construction.','Small2')
    img('accuracy.png',166)
    p('Majority voting reaches 86.22%; first parseable reaches 81.70%, raw candidate zero 78.20%, and uniformly random selection has expected accuracy 75.69%. Statistics and embeddings each reach 89.72%; combining them reaches 92.11%. The selected centered variant reaches 92.23%, only one additional question, so its small advantage over combined LR should not be treated as a robust gain.')
    p('The style-only ablation is weak despite masking invalid answers, which argues against explaining the improvement using length and formatting alone. Nonlinear models and ensembles add assets and fitting cost without improving validation selection. Text-only boosting reaches 90.10% at best. All experiments, including lower-scoring configurations, are recorded in experiments.json.')

    page('3. Reliability, disagreement and split difficulty')
    table(['Distinct valid answers','Questions','Majority %','Selected model %'],[[str(g['distinct_answers']),str(g['n']),f"{100*g['majority_accuracy']:.2f}",f"{100*g['model_accuracy']:.2f}"] for g in A['distinct_answer_groups']],[165,90,110,130])
    p('The 443 validation questions with one distinct valid answer are solved by any rule that selects a parseable candidate. On the 355 informative validation questions with at least two distinct valid answers, accuracy is <b>82.54%</b>, versus <b>69.01%</b> for majority voting. The test bank deliberately contains multiple-answer groups, helping explain its lower 81.37% score. This subset comparison describes a distribution difference; it is not a matched or causal comparison between splits.')
    p('The selected model fixes 60 majority-vote mistakes and introduces 12, a net gain of 48 questions. When the methods select different numeric answers (82 questions), the model is correct on 60, or 73.17%. A paired question bootstrap with 5,000 resamples and seed 2026 gives a validation gain interval of +4.01 to +8.15 percentage points. This is descriptive: it does not account for selecting a model on the same validation bank.')
    img('reliability.png',163)
    p('Confidence is the sigmoid of the selected candidate logit, without post-hoc calibration. Retaining the highest-confidence half of validation gives 99.75% accuracy (398/399), but easy consensus questions inflate this figure. Within the informative subset, the top half reaches 95.51% (170/178). These are retrospective coverage diagnostics, not an abstaining submission or a guarantee on new data.')
    p('The five-bin ECE is 0.0379 and Brier score is 0.0633. The largest bin has mean confidence 0.9786 but accuracy 0.9501, showing overconfidence. Low-confidence bins have only 4, 17 and 14 examples and fluctuate sharply. Confidence is useful for ranking uncertainty here, but should not be interpreted as a well-calibrated probability of valid reasoning.')
    p('Every question must still receive a candidate choice. The evaluation method does not abstain, and no confidence threshold is chosen using test references.','Small2')

    page('4. Failure analysis and limits of verification')
    h('A fluent explanation can confuse people with pairs')
    p('Validation position 44 combines classes with 46 male and 48 female students. There can be 46 pairs, which use 92 students, leaving 2 unpaired. The selected candidate correctly totals the class sizes and finds 46 pairs, but subtracts 46 from 94 to return 48. It confuses a count of pairs with a count of people. Its orderly steps and correct intermediate sums do not make its final inference valid.')
    h('Correct arithmetic can still be applied to the wrong quantities')
    p('At validation position 48, Tom buys 20 shares for $3 each, sells 10 for $4 and values the remaining ten at $6. The reference profit is (40 + 60) - 60 = 40. The selected candidate instead takes the first sale profit of $10, multiplies it by ten again and returns 100, while ignoring the remaining position in its final profit calculation. The selector does not enforce consistent units or cash-flow accounting.')
    h('A local arithmetic slip survives a plausible strategy')
    p('At validation position 52, a dog closes a 0.6-mile gap at 24 - 15 = 9 mph. The selected candidate uses the correct relative-speed method but writes 0.6 / 9 = 2/15 hour, giving 8 minutes. The actual value is 1/15 hour, or 4 minutes. A compact learned verifier can recognize a plausible strategy without detecting its arithmetic error.')
    h('Reference agreement and interpretation can diverge')
    p('At validation position 32, Yun has 8 paperclips. The question says Marion has "1/4 more than what Yun currently has, plus 7." The selected answer is 8 + 2 + 7 = 17, consistent with one ordinary reading of "1/4 more." The supplied reference instead computes 8/4 + 7 = 9. The scorer therefore marks 17 wrong. This example is a possible wording/reference ambiguity, not evidence that all benchmark errors are reasoning failures. The parser and reference were kept unchanged.')
    h('What the experiment does and does not establish')
    p('The evidence supports complementary predictive information in frozen embeddings and inexpensive statistics. It does not establish step-by-step mathematical verification, generalization to a different generator, or causal use of reasoning semantics. The small centered-versus-combined difference, validation search, candidate-level internal MLP split, small hard subgroups and single-machine resource measurements limit stronger conclusions. There is no separate untouched development bank for uncertainty estimates.')
    p('The 62 validation errors, full candidate texts and per-question confidence values are provided in validation_errors.json and validation_choices.csv so that the qualitative analysis can be checked. Future work could add explicit arithmetic or unit-consistency features, but no such change was made after freezing or inspecting the final test score.','Small2')

    page('5. Reproduction, resources and disclosure')
    h('Frozen artifact and evaluation')
    p(f"The frozen model is <b>{E['selected']}</b>. The official test result is <b>{T['correct']}/{T['n']} = {T['accuracy']:.8f}%</b>. The matching assets are code/runs/model_bundle/config.json and lr_context_0.01.npz. runs/freeze.json records SHA-256 hashes and the freeze timestamp, before final test evaluation. Inference needs NumPy only and receives new bank inputs through the official interface; it never opens reference files.")
    p(f"Local hardware: Intel Core i7-13650HX (14 cores / 20 logical processors), Windows 11, Python 3.12.7. Evaluation limits NumPy/BLAS/OpenMP to four threads. Validation prediction plus scoring took {VRES['total_wall_seconds']:.3f} s; test took {TRES['total_wall_seconds']:.3f} s. Inference assets total {F['inference_asset_bytes']:,} bytes, including the uncompressed NPZ container and JSON configuration. All fitted arrays and statistics are included. The full development search took {E['training_total_seconds']:.2f} s locally, including preprocessing. No GPU was used. Reference Xeon runtime was not measured.")
    h('Exact commands from code/ (Python 3.12)')
    code('python -m pip install -r requirements.txt\npython check_data.py\npython evaluate.py --split validation\npython evaluate.py --split test\n\n# Optional retraining in a separate directory; do not overwrite frozen assets\npython -m pip install -r requirements-training.txt\npython train.py --output runs/retrained')
    p('evaluate.py sets all relevant thread environment variables before launching fresh official prediction and scoring processes. The README also lists the direct official commands. Training pins NumPy 1.26.4, SciPy 1.13.1, scikit-learn 1.5.1, joblib 1.4.2 and threadpoolctl 3.5.0. Runs use seeds 2026/2027. Learned scaling is training-only. No train/validation refitting or test-based tuning was performed. The data checker passed all hashes and alignment checks; inference was checked on renamed IDs, a seven-question subset and permuted candidates.')
    h('AI assistance and reused material')
    p('OpenAI Codex provided substantive assistance with feature/model design, implementation, debugging, local experiment execution, analysis and report drafting. This assistance extends beyond code completion. The included experiment files are actual recorded outputs. The submitter should review and understand the implementation and analysis before submission. Official loader, numeric parser, runner and independent scorer are reused unchanged; no other external solution code, datasets or pretrained weights are used.')
    h('Sources and attribution')
    p('[1] DASE7506-MP2 Project Guide and running instructions, upstream commit bb801f4f2e7629d270df765ab12c333b2dc616bb.<br/>github.com/xudongwu-0/DASE7506-MP2<br/>[2] GSM8K: github.com/openai/grade-school-math (MIT; supplied license retained).<br/>[3] Supplied representations/candidates: Qwen3-0.6B; source revision recorded in code/data/manifest.json.<br/>[4] scikit-learn implementations of LogisticRegression, MLPClassifier and HistGradientBoostingClassifier, version 1.5.1.','Small2')
    p('The result and files are ready for repository publication. Creating the required GitHub submission issue is a separate final action; this report does not claim that submission has occurred.','Small2')

    def chrome(canvas,doc):
        canvas.saveState(); w,h=A4
        canvas.setStrokeColor(colors.HexColor('#D4E0E6')); canvas.line(50,h-35,w-50,h-35)
        canvas.setFillColor(colors.HexColor('#536579')); canvas.setFont('Helvetica',8)
        canvas.drawString(50,h-27,'DASE7506  /  SELECTING RELIABLE LLM REASONING')
        canvas.line(50,36,w-50,36); canvas.drawString(50,24,'Frozen result: 380 / 467  |  31 October 2026'); canvas.drawRightString(w-50,24,str(doc.page)); canvas.restoreState()
    doc=SimpleDocTemplate(str(R/'report.pdf'),pagesize=A4,rightMargin=50,leftMargin=50,topMargin=53,bottomMargin=51,title='Selecting Reliable LLM Reasoning',author='DASE7506 MP2 coursework')
    doc.build(story,onFirstPage=chrome,onLaterPages=chrome)
    (R/'report.md').write_text('\n\n'.join(md)+'\n',encoding='utf-8')
    print(R/'report.pdf')

if __name__=='__main__':
    if '--figures-only' in sys.argv: figures()
    else: build()
