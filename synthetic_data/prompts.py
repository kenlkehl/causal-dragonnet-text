# synthetic_data/prompts.py
"""LLM prompt templates for synthetic data generation."""

# System prompt for clinical expertise
CLINICAL_SYSTEM_PROMPT = """You are an expert clinical researcher and oncologist with deep expertise in:
- Comparative effectiveness research
- Clinical trial design
- Real-world evidence generation
- Causal inference methodology

You provide precise, structured responses that can be parsed programmatically.
Always respond with valid JSON when requested."""


# =============================================================================
# Confounder Generation Prompt
# =============================================================================

CONFOUNDER_GENERATION_PROMPT = """Given the following comparative effectiveness research question:

"{clinical_question}"

Generate a comprehensive list of realistic confounding variables that would influence both treatment assignment and outcome in a real-world clinical setting.

Requirements:
1. Include 8-12 confounders total
2. Mix of categorical (3-5 categories each) and continuous variables
3. Common confounders might include: age, sex (if applicable to the cancer type), performance status, comorbidities, prior treatments, biomarkers, disease stage, etc.
4. Be specific to the clinical context of the question

Respond with a JSON object in this exact format:
{{
  "confounders": [
    {{
      "name": "age",
      "type": "continuous",
      "description": "Patient age in years"
    }},
    {{
      "name": "ecog_performance_status",
      "type": "categorical",
      "categories": ["0", "1", "2", "3"],
      "description": "ECOG performance status score"
    }},
    ...
  ]
}}"""


# =============================================================================
# Regression Equation Generation Prompt
# =============================================================================

REGRESSION_EQUATION_PROMPT = """Given the following confounding variables for a comparative effectiveness study:

Clinical Question: "{clinical_question}"

Confounders:
{confounder_list}

Generate two plausible regression equations for a simulation:

1. TREATMENT ASSIGNMENT equation: Predicts logit(P(treatment=1)) based on confounders
   - Should reflect realistic clinical decision-making
   - Some confounders should have stronger effects than others
   - Include 2-3 interaction terms between clinically related confounders

2. OUTCOME equation: Predicts logit(P(outcome=1)) based on confounders AND treatment
   - The treatment coefficient is FIXED at {treatment_coefficient} (do not include treatment in your coefficients)
   - Should reflect known prognostic factors
   - Include 2-3 interaction terms
   - Some confounders may affect outcome differently than treatment assignment

For continuous variables, coefficients represent effect per 1 SD increase.
For categorical variables, coefficients are relative to the reference category (first listed).

Respond with JSON in this exact format:
{{
  "treatment_equation": {{
    "intercept": -0.5,
    "coefficients": {{
      "age": 0.3,
      "ecog_performance_status_1": -0.2,
      "ecog_performance_status_2": -0.5,
      "ecog_performance_status_3": -0.8
    }},
    "interactions": [
      {{
        "terms": ["age", "comorbidity_count"],
        "coefficient": 0.1
      }}
    ]
  }},
  "outcome_equation": {{
    "intercept": -1.0,
    "coefficients": {{
      "age": -0.2,
      "ecog_performance_status_1": 0.3,
      "ecog_performance_status_2": 0.6,
      "ecog_performance_status_3": 1.2
    }},
    "interactions": [
      {{
        "terms": ["age", "prior_treatment"],
        "coefficient": -0.15
      }}
    ]
  }}
}}

Note: For categorical variables with N categories, create N-1 dummy variable coefficients (excluding the reference/first category).
The coefficient names for dummies should be: variablename_categoryvalue"""


# =============================================================================
# Summary Statistics Generation Prompt
# =============================================================================

SUMMARY_STATISTICS_PROMPT = """Given the following confounding variables for a study on:

"{clinical_question}"

Confounders:
{confounder_list}

Generate realistic summary statistics that would be observed in a real-world patient population.

For each confounder, provide:
- Categorical variables: proportion of patients in each category (must sum to 1.0)
- Continuous variables: mean and standard deviation

Base these on realistic clinical populations. For example:
- Age distributions typical for the cancer type
- Performance status distributions reflecting real-world (sicker than trials)
- Comorbidity rates appropriate for the demographic

Respond with JSON in this exact format:
{{
  "summary_statistics": {{
    "age": {{
      "type": "continuous",
      "mean": 65.0,
      "std": 12.0
    }},
    "ecog_performance_status": {{
      "type": "categorical",
      "proportions": {{
        "0": 0.25,
        "1": 0.45,
        "2": 0.20,
        "3": 0.10
      }}
    }},
    ...
  }}
}}"""


# =============================================================================
# Patient History Generation Prompt  
# =============================================================================

PATIENT_HISTORY_PROMPT = """You are generating a realistic synthetic clinical history document for a cancer patient.
This document should simulate concatenated clinical notes, radiology reports, and pathology reports.

Patient Characteristics:
{patient_characteristics}

Clinical Context: {clinical_question}

Generate a comprehensive clinical history document that:
1. Simulating at least 5 clinical documents, concatenated together
2. Includes sections from different note types:
   - Initial oncology consultation note
   - At least one radiology report (CT, PET, or MRI)
   - At least one pathology report
   - One or more follow-up notes
3. Naturally incorporates ALL the patient characteristics listed above
4. Uses realistic medical terminology and abbreviations
5. Includes dates (use relative dates like "3 months prior", "at diagnosis")
6. Contains typical clinical details like vital signs, lab values, physical exam findings
7. Reflects the clinical decision-making around treatment selection

Important: The patient characteristics should be embedded naturally in the clinical narrative, not listed explicitly.
Write the document as if it were real clinical notes, imaging reports, and pathology reports that have been concatenated together.

Begin the clinical history document now:"""


def format_confounder_list(confounders: list) -> str:
    """Format confounders into a readable list for prompts."""
    lines = []
    for c in confounders:
        if c["type"] == "categorical":
            cats = ", ".join(c["categories"])
            lines.append(f"- {c['name']} (categorical): {c['description']}. Categories: [{cats}]")
        else:
            lines.append(f"- {c['name']} (continuous): {c['description']}")
    return "\n".join(lines)


def format_patient_characteristics(characteristics: dict, confounders: list) -> str:
    """Format patient characteristics into readable text for history generation."""
    lines = []
    confounder_map = {c["name"]: c for c in confounders}
    
    for name, value in characteristics.items():
        conf = confounder_map.get(name, {})
        desc = conf.get("description", name.replace("_", " ").title())
        
        if conf.get("type") == "continuous":
            # Format continuous with units if known
            if "age" in name.lower():
                lines.append(f"- {desc}: {value:.0f} years")
            else:
                lines.append(f"- {desc}: {value:.2f}")
        else:
            lines.append(f"- {desc}: {value}")
    
    return "\n".join(lines)
