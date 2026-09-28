# B2B Opportunity Prioritization Tool

Academic deployment prototype for the AADM B2B Sales & Marketing Analytics project.

## What it deploys
- Logistic Regression: conversion probability
- Linear Regression: estimated Potential Deal Size
- K-Means: opportunity segment

## Models
The saved models were retrained using the final notebook variable definitions, preprocessing and random states.

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open the local Streamlit URL shown in the terminal.

## Important modeling note
The conversion model intentionally excludes downstream/customer-outcome variables such as Customer Lifetime Value, Customer Satisfaction, Renewal Likelihood, Customer Acquisition Cost and Marketing ROI, as well as SalesStage, to avoid inappropriate temporal use and target leakage.

## Deployment explanation for viva
The deployment operationalizes the trained analytical models as an interactive decision-support tool. A user enters a new B2B opportunity's available characteristics and receives a model-based conversion probability, estimated opportunity value and historical opportunity segment. The tool is not an autonomous decision-maker and does not imply causality.
