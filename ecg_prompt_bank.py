"""Starter LLM-style prompt ensemble for PTB-XL diagnostic superclasses.

BiomedCoOp uses many class-specific prompts (50 in the official configs).
This module creates 50 prompts per class from 10 clinically oriented
descriptions and 5 ECG-specific templates.

For a paper-quality experiment, you can replace CLASS_FINDINGS with prompts
generated independently by an LLM and reviewed by an ECG expert.
"""

from __future__ import annotations

from typing import Dict, List


PROMPT_TEMPLATES = [
    "A 12-lead electrocardiogram demonstrating {}.",
    "An ECG tracing with {}.",
    "This electrocardiogram is characterized by {}.",
    "The cardiac electrical pattern suggests {}.",
    "The ECG findings are consistent with {}.",
]


CLASS_FINDINGS: Dict[str, List[str]] = {
    "NORM": [
        "normal sinus rhythm with normal conduction intervals and no significant ST-T abnormality",
        "a normal 12-lead ECG without evidence of acute ischemia or conduction block",
        "normal cardiac electrical activity with regular rhythm and expected waveform morphology",
        "a physiologic ECG pattern without diagnostic abnormalities",
        "normal atrial and ventricular depolarization and repolarization",
        "a regular sinus rhythm with normal QRS duration and normal repolarization",
        "an ECG within normal limits for rate, rhythm, axis, intervals, and morphology",
        "no electrocardiographic evidence of infarction, hypertrophy, or conduction disturbance",
        "normal P-wave, QRS-complex, ST-segment, and T-wave morphology",
        "a normal diagnostic ECG pattern without clinically significant abnormalities",
    ],
    "MI": [
        "electrocardiographic evidence compatible with myocardial infarction",
        "an infarction-related ECG pattern with pathological Q waves or ischemic ST-T changes",
        "findings suggesting acute or prior myocardial infarction",
        "a myocardial injury pattern localized by abnormal QRS and repolarization morphology",
        "an ECG pattern consistent with ischemic myocardial necrosis",
        "infarction-associated abnormalities affecting ventricular depolarization or repolarization",
        "diagnostic features that support myocardial infarction",
        "an abnormal ECG suggestive of previous or ongoing myocardial infarction",
        "Q-wave or ST-T findings compatible with infarcted myocardium",
        "a clinically significant myocardial infarction pattern",
    ],
    "STTC": [
        "ST-segment and T-wave abnormalities consistent with altered ventricular repolarization",
        "non-normal ST-T morphology indicating a repolarization disturbance",
        "ST-segment deviation or abnormal T-wave morphology",
        "a ventricular repolarization abnormality involving the ST segment or T wave",
        "diffuse or regional ST-T wave changes",
        "an ECG pattern dominated by abnormal ventricular repolarization",
        "clinically significant ST-segment or T-wave changes",
        "repolarization changes without requiring a specific infarction diagnosis",
        "abnormal ST-T configuration across one or more ECG leads",
        "an ST/T-wave change diagnostic superclass pattern",
    ],
    "CD": [
        "an intraventricular or atrioventricular conduction disturbance",
        "delayed or abnormal cardiac impulse conduction",
        "a bundle-branch, fascicular, or atrioventricular conduction abnormality",
        "abnormal QRS conduction morphology or prolonged conduction intervals",
        "an ECG pattern consistent with conduction system disease",
        "impaired propagation of electrical activity through the cardiac conduction system",
        "a widened or morphologically abnormal QRS complex due to conduction disturbance",
        "electrocardiographic evidence of a conduction block or delay",
        "a clinically significant cardiac conduction abnormality",
        "a conduction-disturbance diagnostic superclass pattern",
    ],
    "HYP": [
        "voltage and morphology findings compatible with cardiac chamber hypertrophy",
        "an ECG pattern suggesting ventricular or atrial hypertrophy",
        "increased cardiac electrical forces consistent with myocardial hypertrophy",
        "hypertrophy-associated voltage criteria with possible secondary repolarization changes",
        "electrocardiographic evidence of enlarged or hypertrophied cardiac chambers",
        "a high-voltage QRS pattern compatible with ventricular hypertrophy",
        "a chamber-enlargement pattern affecting atrial or ventricular waveforms",
        "findings supporting left-sided or right-sided cardiac hypertrophy",
        "an abnormal ECG morphology caused by increased myocardial mass",
        "a hypertrophy diagnostic superclass pattern",
    ],
}


def build_teacher_prompt_bank() -> Dict[str, List[str]]:
    """Return exactly 50 prompts for each PTB-XL superclass."""
    bank: Dict[str, List[str]] = {}
    for class_name, findings in CLASS_FINDINGS.items():
        prompts = [
            template.format(finding)
            for finding in findings
            for template in PROMPT_TEMPLATES
        ]
        if len(prompts) != 50:
            raise RuntimeError(
                f"{class_name} generated {len(prompts)} prompts; expected 50"
            )
        bank[class_name] = prompts
    return bank
