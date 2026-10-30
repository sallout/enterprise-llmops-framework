from pathlib import Path
import re
p=Path('output/dissertation/final_dissertation.md');s=p.read_text()
a=s.index('# 2 Related work');b=s.index('## 2.2 Coding instruction',a)
s=s[:a]+'''# 2 Related work and research scope

## 2.1 Enterprise adaptation and the research gap

The project addresses two connected requirements: programming assistance and answers about a limited set of corporate rules. A public coding dataset supplies examples of programming tasks, while corporate documents and repository sources provide organisation-specific material. Their different origins require explicit source metadata, compatible conversation records and separate evidence about correctness.

The research questions concern whether these inputs can be prepared reproducibly, whether limited structural reduction can be followed by recovery training and a reloadable 4-bit export, and how the resulting model performs against the original Qwen3-14B. The current implementation provides processing and execution evidence. Independent coding and policy measurements are still required to answer the performance questions.

The literature below supports the choice of instruction data, adaptation methods and evaluation criteria. Published outcomes are attributed to their original studies; their transfer to this project's data and model remains an experimental question.

'''+s[b:]
s=s.replace(' Repository-level editing adds a further requirement: the complete changed repository must satisfy an appropriate oracle, as CodePlan illustrates [4].','')
s=s.replace("Context extraction is bounded and heuristic, so it does not provide CodePlan's repository-wide semantic relations.","Context extraction is bounded and heuristic; dependencies outside the selected context may be omitted.")
a=s.index('## 9.2 A CodePlan-inspired extension');b=s.index('Completion note P1:',a)
s=s[:a]+'''## 9.2 Experimental evidence and publication scope

A publishable account should state the research questions, describe the data selection and model changes precisely, and report each measurement with its experimental conditions. The completed short experiment demonstrates execution through pruning, recovery and NF4 export. The larger run and independent comparison must supply the remaining evidence about coding and enterprise behaviour.

The comparison should keep the evaluation tasks, prompt information and generation settings fixed across the original and resulting models. Report functional code-test outcomes separately from policy-answer scores and efficiency measurements. If the study attributes an improvement to a particular training source, matched source-ablation experiments are required. A mixed-data run alone cannot identify that source's contribution.

The practical contribution is the implemented data-to-model workflow and its measured outcomes within the stated scope. Tables of pending results must be replaced with observations before submission or publication. Failed and excluded cases, incomplete upstream sources, sampling limitations and the small corporate holdout should remain visible when interpreting those outcomes.

## 9.3 Release requirements

'''+s[b:]
s=s.replace('remaining corporate-response and planning-interface behaviour','remaining corporate-response behaviour')
s=re.sub(r'^\[4\] R\. Bairi.*\n\n','',s,flags=re.M)
s=re.sub(r'^\| CodePlan \|.*\n','',s,flags=re.M)
s=re.sub(r'^\| Oracle \|.*\n','',s,flags=re.M)
def renumber(m):
    nums=[int(x) for x in re.findall(r'\d+',m.group(1))]
    assert 4 not in nums
    return '['+', '.join(str(n-1 if n>4 else n) for n in nums)+']'
s=re.sub(r'\[([0-9, ]+)\]',renumber,s)
assert 'codeplan' not in s.lower()
p.write_text(s)
b=Path('tmp/dissertation_build/build_report.py');t=b.read_text()
t=t.replace('CodePlan informs the discussion of repository context and correctness oracles, while adaptive repository editing remains a proposed application extension. ','')
t=t.replace('; quantization; CodePlan', '; quantization')
b.write_text(t)
