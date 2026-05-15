#!/usr/bin/env python3
"""
Generate 2000 synthetic clinical notes per hospital (3 hospitals).
Each hospital has a distinct specialty distribution to simulate real non-IID data.
Notes are structured like real EHR entries with realistic variation.
"""

import json
import random
import os

random.seed(42)

# ======================================================================
# Templates and vocabulary per specialty
# ======================================================================

CARDIO_TEMPLATES = [
    "{age}{sex} presents with {cc}. {vitals}. {labs}. {imaging}. {plan}.",
    "{age}{sex} admitted for {cc}. {hpi} {vitals}. {labs}. {assessment}. {plan}.",
    "{age}{sex} with PMH of {pmh}, presenting with {cc}. {vitals}. {labs}. {plan}.",
]

PULM_TEMPLATES = [
    "{age}{sex} with {cc}. {vitals}. {labs}. {imaging}. {plan}.",
    "{age}{sex} admitted for {cc}. {hpi} {vitals}. PFTs: {pfts}. {imaging}. {plan}.",
    "{age}{sex} with PMH of {pmh}, presenting with {cc}. {vitals}. {labs}. {plan}.",
]

SEPSIS_TEMPLATES = [
    "{age}{sex} brought to ED with {cc}. {vitals}. {labs}. SOFA score {sofa}. {cultures}. {plan}.",
    "{age}{sex} from {source} with {cc}. {vitals}. Lactate {lactate}. {labs}. {cultures}. {plan}.",
    "{age}{sex} with PMH of {pmh}, presenting with {cc}. {vitals}. {labs}. qSOFA {qsofa}/3. {plan}.",
]

# --- Cardiology vocabulary ---
CARDIO_CC = [
    "acute chest pain radiating to left arm", "new-onset dyspnea on exertion",
    "palpitations and near-syncope", "acute decompensated heart failure",
    "STEMI with ST elevation in V1-V4", "NSTEMI with troponin rise",
    "aortic stenosis with syncope", "hypertensive emergency BP 220/120",
    "acute pulmonary edema", "ventricular tachycardia storm",
    "bradycardia with complete heart block", "infective endocarditis",
    "acute aortic dissection Type A", "cardiac tamponade",
    "worsening dyspnea with orthopnea", "atrial fibrillation with RVR",
]
CARDIO_LABS = [
    "Troponin 4.2 ng/mL, CK-MB 38", "BNP 2400 pg/mL, Cr 1.8",
    "Troponin 0.8 trending up, K+ 5.2", "BNP 890, pro-BNP 4200",
    "INR 2.8, PTT 42", "Troponin negative x2, D-dimer 1.2",
    "Troponin 12.4, CK 840", "BNP 3600, Na 128, Cr 2.4",
    "Lipid panel: LDL 182, HDL 32", "HbA1c 9.2, fasting glucose 242",
]
CARDIO_IMAGING = [
    "Echo: EF 25%, severe MR, dilated LV", "Echo: EF 55%, moderate AS, valve area 0.9cm2",
    "CTA: Type A dissection extending to iliac", "Echo: large pericardial effusion with tamponade",
    "Echo: EF 35%, akinetic anterior wall", "Cath: 95% LAD stenosis, 80% RCA",
    "Echo: EF 60%, severe TR, RVSP 65", "CXR: bilateral pleural effusions, cardiomegaly",
    "Echo: EF 15%, global hypokinesis", "Nuclear stress: reversible anterior defect",
]
CARDIO_PLAN = [
    "Emergent PCI with DES to LAD. Dual antiplatelet. Statin. Beta-blocker",
    "IV furosemide 80mg, dobutamine drip. Cardiology consult for advanced HF therapies",
    "Rate control with IV diltiazem. Anticoagulation with heparin bridge to warfarin",
    "TAVR evaluation. Optimize volume status. Diuresis with bumetanide",
    "ICD implantation for primary prevention. Amiodarone load. EP consult",
    "Fibrinolytic therapy with alteplase. Transfer for emergent CABG",
    "Pericardiocentesis performed, 450mL drained. Hemodynamics improved",
    "Started carvedilol, lisinopril, spironolactone. Fluid restriction 1.5L",
    "Cardioversion performed, NSR restored. Started flecainide maintenance",
    "IV nitroglycerin and esmolol for dissection. CT surgery consult",
]

# --- Pulmonology vocabulary ---
PULM_CC = [
    "acute COPD exacerbation with respiratory failure", "massive hemoptysis",
    "bilateral pulmonary embolism", "pneumothorax with respiratory distress",
    "severe asthma exacerbation unresponsive to bronchodilators",
    "new lung mass on imaging", "progressive dyspnea over 6 months",
    "interstitial lung disease with declining PFTs", "pleural effusion with dyspnea",
    "post-transplant rejection", "acute respiratory failure requiring intubation",
    "obstructive sleep apnea with cor pulmonale", "sarcoidosis with hilar lymphadenopathy",
    "aspiration pneumonia", "pulmonary hypertension NYHA III",
    "lung abscess with empyema",
]
PULM_PFTS = [
    "FEV1 28%, FVC 52%, ratio 0.42, DLCO 35%", "FEV1 65%, FVC 70%, ratio 0.73, DLCO 48%",
    "FEV1 82%, FVC 60%, ratio 1.08, DLCO 42% — restrictive", "FEV1 45%, FVC 50%, ratio 0.71",
    "FEV1 92%, FVC 94%, normal spirometry, DLCO 38%", "FEV1 22%, FVC 40%, severe obstruction",
]
PULM_IMAGING = [
    "CT: 4.5cm RUL mass with mediastinal lymphadenopathy",
    "CTPA: saddle PE with RV strain", "CXR: bilateral infiltrates with air bronchograms",
    "HRCT: UIP pattern with honeycombing", "CT: large right pneumothorax with mediastinal shift",
    "CT: bilateral ground-glass opacities", "CXR: large left pleural effusion",
    "PET: FDG-avid RUL mass, SUV 12.4", "CT: traction bronchiectasis, fibrotic NSIP",
    "V/Q scan: high probability for PE", "CT: cavitary lesion RLL with air-fluid level",
]
PULM_PLAN = [
    "BiPAP, IV methylprednisolone 125mg, azithromycin, ceftriaxone",
    "Heparin drip, consider TPA given RV strain. IVC filter consult",
    "Chest tube placed, lung re-expanded. Pleurodesis if recurrent",
    "Started nintedanib 150mg BID. Pulmonary rehab referral. O2 2L NC",
    "Continuous albuterol, IV magnesium 2g, methylpred. ICU admission",
    "Bronchoscopy with biopsy scheduled. Staging CT abdomen/pelvis",
    "Thoracentesis: 1.8L exudative. Cytology sent. Pleurodesis planned",
    "Inhaled treprostinil started. Right heart cath shows mPAP 48",
    "Started prednisone 40mg taper. Methotrexate if no response",
    "IR-guided drainage of abscess. IV ampicillin-sulbactam",
]

# --- Sepsis vocabulary ---
SEPSIS_CC = [
    "fever, hypotension, and altered mental status", "septic shock from urinary source",
    "bacteremia with hemodynamic instability", "necrotizing soft tissue infection",
    "perforated viscus with peritonitis", "pneumonia with sepsis",
    "cholangitis with septic shock", "C. difficile colitis with toxic megacolon",
    "meningitis with septic shock", "infected prosthetic joint",
    "post-operative wound infection with bacteremia", "neutropenic fever with sepsis",
    "endocarditis with septic emboli", "intra-abdominal abscess",
    "catheter-related bloodstream infection", "spontaneous bacterial peritonitis",
]
SEPSIS_CULTURES = [
    "Blood cx: E. coli, pan-sensitive", "Blood cx: MRSA",
    "Blood cx: Klebsiella pneumoniae ESBL+", "Urine cx: Pseudomonas aeruginosa",
    "Blood cx: Strep pneumoniae", "Blood cx: Enterococcus faecalis",
    "Wound cx: polymicrobial including anaerobes", "Blood cx: Staph aureus MSSA",
    "CSF cx: Neisseria meningitidis", "Blood cx: Candida albicans",
    "Sputum cx: MRSA", "Blood cx: Bacteroides fragilis",
    "Blood cx pending, started empiric broad-spectrum", "Blood cx: Strep pyogenes",
]
SEPSIS_PLAN = [
    "30mL/kg crystalloid bolus. Piperacillin-tazobactam 4.5g IV q6h. Norepinephrine started",
    "Meropenem 1g IV q8h. Vasopressin added. Stress dose hydrocortisone 50mg q8h",
    "Vancomycin 25mg/kg load + meropenem. Central line placed. A-line for continuous BP",
    "Emergent surgical debridement. Linezolid + meropenem + clindamycin",
    "ERCP for biliary decompression. Piperacillin-tazobactam. Aggressive fluid resuscitation",
    "Ceftriaxone 2g + vancomycin 25mg/kg + dexamethasone. Lumbar puncture performed",
    "Line removal. Micafungin 100mg IV daily. Repeat blood cultures q48h",
    "Oral vancomycin 125mg QID + IV metronidazole. Surgery consult for toxic megacolon",
    "Nephrostomy tube placed. Gentamicin + ampicillin. Urology follow-up",
    "Cefepime 2g IV q8h. G-CSF for neutropenic fever. Oncology consulted",
]

PMH_OPTIONS = [
    "HTN, DM2, CKD3", "CAD s/p CABG, CHF EF 30%, AFIB",
    "COPD on home O2, former smoker 40py", "DM1, ESRD on HD, HTN",
    "Obesity BMI 42, OSA on CPAP, HTN", "cirrhosis Child-Pugh B, portal HTN",
    "lung cancer s/p lobectomy, COPD", "mechanical aortic valve on warfarin",
    "rheumatoid arthritis on methotrexate", "HIV on ART, CD4 350",
    "SLE on hydroxychloroquine", "prior DVT/PE on rivaroxaban",
    "transplant on tacrolimus", "atrial fibrillation on apixaban",
    "chronic kidney disease stage 4", "type 2 diabetes on insulin",
]

SOURCES = [
    "nursing facility", "home", "outside hospital transfer",
    "rehabilitation center", "group home", "EMS from scene",
]


def rand_vitals():
    hr = random.randint(55, 150)
    sbp = random.randint(70, 200)
    dbp = random.randint(40, 110)
    rr = random.randint(12, 36)
    temp = round(random.uniform(36.0, 40.5), 1)
    spo2 = random.randint(78, 100)
    return f"VS: HR {hr}, BP {sbp}/{dbp}, RR {rr}, T {temp}C, SpO2 {spo2}%"


def rand_labs():
    wbc = round(random.uniform(1.2, 28.0), 1)
    hgb = round(random.uniform(6.5, 16.0), 1)
    plt = random.randint(40, 450)
    na = random.randint(125, 150)
    k = round(random.uniform(2.8, 6.2), 1)
    cr = round(random.uniform(0.6, 5.8), 1)
    return f"WBC {wbc}K, Hgb {hgb}, Plt {plt}K, Na {na}, K {k}, Cr {cr}"


def gen_cardio_note():
    return random.choice(CARDIO_TEMPLATES).format(
        age=random.randint(35, 92), sex=random.choice(["M", "F"]),
        cc=random.choice(CARDIO_CC), vitals=rand_vitals(),
        labs=random.choice(CARDIO_LABS), imaging=random.choice(CARDIO_IMAGING),
        plan=random.choice(CARDIO_PLAN), hpi=f"Onset {random.choice(['acute','gradual','sudden'])} {random.randint(1,72)} hours ago.",
        pmh=random.choice(PMH_OPTIONS), assessment=f"Assessment: {random.choice(CARDIO_CC)}",
    )


def gen_pulm_note():
    return random.choice(PULM_TEMPLATES).format(
        age=random.randint(25, 88), sex=random.choice(["M", "F"]),
        cc=random.choice(PULM_CC), vitals=rand_vitals(),
        labs=rand_labs(), imaging=random.choice(PULM_IMAGING),
        plan=random.choice(PULM_PLAN), pfts=random.choice(PULM_PFTS),
        hpi=f"Symptoms worsening over {random.randint(1,14)} days.",
        pmh=random.choice(PMH_OPTIONS),
    )


def gen_sepsis_note():
    return random.choice(SEPSIS_TEMPLATES).format(
        age=random.randint(28, 95), sex=random.choice(["M", "F"]),
        cc=random.choice(SEPSIS_CC), vitals=rand_vitals(),
        labs=rand_labs(), cultures=random.choice(SEPSIS_CULTURES),
        plan=random.choice(SEPSIS_PLAN), lactate=round(random.uniform(1.0, 12.0), 1),
        sofa=random.randint(2, 18), qsofa=random.randint(1, 3),
        source=random.choice(SOURCES), pmh=random.choice(PMH_OPTIONS),
        hpi=f"Found {random.choice(['unresponsive','febrile','hypotensive'])} at {random.choice(SOURCES)}.",
    )


def main():
    N = 2000
    os.makedirs("clinical_data", exist_ok=True)

    generators = {
        0: ("City Heart Center", gen_cardio_note),
        1: ("Regional Lung Institute", gen_pulm_note),
        2: ("Metro Emergency Hospital", gen_sepsis_note),
    }

    # Also mix in 15% cross-specialty for realism
    all_gens = [gen_cardio_note, gen_pulm_note, gen_sepsis_note]

    for hid, (name, primary_gen) in generators.items():
        notes = []
        for i in range(N):
            if random.random() < 0.85:
                notes.append(primary_gen())
            else:
                notes.append(random.choice(all_gens)())

        path = f"clinical_data/hospital_{hid}.json"
        with open(path, "w") as f:
            json.dump({"name": name, "notes": notes}, f, indent=2)
        print(f"Hospital {hid} ({name}): {len(notes)} notes → {path}")

    # Non-member notes for MIA testing
    nonmember = []
    for _ in range(500):
        nonmember.append(random.choice(all_gens)())
    with open("clinical_data/nonmember.json", "w") as f:
        json.dump({"notes": nonmember}, f, indent=2)
    print(f"Non-member: {len(nonmember)} notes → clinical_data/nonmember.json")


if __name__ == "__main__":
    main()
