import pandas as pd
import numpy as np
import json
from datetime import datetime

np.random.seed(42)  # For reproducibility

# Define NE states for location
ne_states = ['Assam', 'Arunachal Pradesh', 'Manipur', 'Meghalaya', 'Mizoram', 'Nagaland', 'Sikkim', 'Tripura']

# Project types
project_types = ['Road', 'Power', 'Tourism', 'Health', 'Education']

# Generate 500 synthetic DPR records
n_samples = 500
data = []

for i in range(n_samples):
    # Metadata
    project_type = np.random.choice(project_types)
    location = np.random.choice(ne_states)
    funding_amount = np.random.uniform(1, 100, 1)[0]  # in Cr
    prior_experience = np.random.choice([0, 1, 2, 3, 5, 10])  # years
    
    # Estimated duration
    estimated_duration = np.random.uniform(6, 48, 1)[0]  # months
    
    # Outcomes (with light correlations: low experience → higher overrun/delay)
    overrun_factor = 1.5 if prior_experience < 2 else 1.0
    cost_overrun_pct = np.random.uniform(0, 50 * overrun_factor, 1)[0] if np.random.rand() > 0.3 else 0
    delay_prob = 0.6 if prior_experience < 2 else 0.4
    delay = np.random.choice([True, False], p=[delay_prob, 1 - delay_prob])
    actual_duration = estimated_duration * (1.2 if delay else 1) + np.random.uniform(-2, 5, 1)[0]
    approval_outcome = np.random.choice(['Approved', 'Rejected', 'Revised'], p=[0.7, 0.1, 0.2])
    
    # Issues found (random 0-3 issues, more if poor quality)
    possible_issues = ['Budget Miscalculation', 'Environmental Non-Compliance', 'Timeline Unrealistic', 'Technical Flaw']
    num_issues = np.random.randint(0, 4)
    issues = np.random.choice(possible_issues, size=num_issues, replace=False).tolist()
    issues_found = '; '.join(issues) if issues else 'None'
    
    # Quality score and category (correlated with metadata: higher funding → slightly better score)
    quality_score = np.random.uniform(4, 10, 1)[0] + (funding_amount / 100) * 0.5  # Slight boost for larger projects
    quality_score = min(10, quality_score)  # Cap at 10
    if quality_score >= 7:
        quality_category = 'Good'
    elif quality_score >= 5:
        quality_category = 'Needs Revision'
    else:
        quality_category = 'Poor'
    
    # Mock DPR sections with simple text (for extraction simulation)
    sections = {
        'Introduction': f'Project for {project_type} in {location}. Estimated cost: {funding_amount:.1f} Cr.',
        'Cost Estimate': f'Total budget: {funding_amount:.1f} Cr. Breakdown: Materials {funding_amount*0.4:.1f}, Labor {funding_amount*0.3:.1f}, etc.',
        'Timeline': f'Estimated duration: {estimated_duration:.1f} months. Start: {datetime.now().strftime("%Y-%m-%d")}.',
        'Environmental Impact': f'EIA {"included" if np.random.rand()>0.3 else "not mentioned"}. Risks: {"Low" if quality_score>7 else "High"}.',
        'Technical Specs': f'Specs for {project_type}: Standard guidelines followed {"with issues" if quality_score<6 else ""}.'
    }
    
    # For annotation simulation: simple offsets (mock, assuming concatenated text)
    full_text = ' '.join(sections.values())
    mock_annotations = []
    # Add section annotation example
    cost_start = full_text.lower().find('cost')
    if cost_start != -1:
        mock_annotations.append({'section': 'Cost', 'start': cost_start, 'end': cost_start + 10})
    # Add entity example (budget)
    budget_str = str(round(funding_amount,1))
    budget_start = full_text.find(budget_str)
    if budget_start != -1:
        mock_annotations.append({'entity': 'Budget', 'start': budget_start, 'end': budget_start + len(budget_str)})
    
    record = {
        'id': i+1,
        'project_type': project_type,
        'location': location,
        'funding_amount_cr': round(funding_amount, 2),
        'prior_experience_years': prior_experience,
        'estimated_duration_months': round(estimated_duration, 1),
        'actual_duration_months': round(actual_duration, 1),
        'cost_overrun_pct': round(cost_overrun_pct, 2),
        'delay': delay,
        'approval_outcome': approval_outcome,
        'issues_found': issues_found,
        'quality_score': round(quality_score, 2),
        'quality_category': quality_category,
        'dpr_sections': sections,
        'full_dpr_text': full_text,
        'annotations': mock_annotations
    }
    data.append(record)

# Create DataFrame
df = pd.DataFrame(data)

# Save to CSV (main tabular data for outcomes/metadata)
df_metadata_outcomes = df[['id', 'project_type', 'location', 'funding_amount_cr', 'prior_experience_years',
                           'estimated_duration_months', 'actual_duration_months', 'cost_overrun_pct', 'delay',
                           'approval_outcome', 'issues_found', 'quality_score', 'quality_category']].copy()
df_metadata_outcomes.to_csv('synthetic_dpr_metadata_outcomes.csv', index=False)

# Save DPR texts and annotations as JSON for extraction/compliance training
dpr_json = df[['id', 'full_dpr_text', 'dpr_sections', 'annotations']].to_dict('records')
with open('synthetic_dpr_texts_annotations.json', 'w') as f:
    json.dump(dpr_json, f, indent=2)

print("Dataset generated successfully!")
print(f"Total samples: {len(df)}")