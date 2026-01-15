# NuPlan Cross-City Generalization

## Cross-City Generalization and Uncertainty Failures in End-to-End Motion Planners (NuPlan)

### Motivation

End-to-end (E2E) motion planners such as NuPlan’s MLPlanner achieve strong performance when evaluated in the same city they were trained on (Boston → Boston).
But performance collapses under distribution shift:

* different road topology
* different traffic conventions
* different multi-agent behavior
* left-hand vs right-hand driving

This project builds a diagnostic pipeline to study cross-city failures and uncertainty miscalibration in E2E planners.

### Objectives

* Quantify cross-city generalization gaps (Boston ↔ Singapore/Pittsburgh).
* Build a scenario-level failure taxonomy.
* Analyze uncertainty (MC Dropout) and calibration (ECE, Brier Score).
* Produce visual diagnostics and interactive failure summaries.

### Research Questions

* RQ1 Cross-City Failure Modes
How does MLPlanner degrade when transferred across cities with different road structures and driving cultures?

* RQ2 Scenario-Specific Failures
Which scenario categories (turns, occlusions, multi-agent conflict, roundabouts) cause the sharpest errors?

* RQ3 Uncertainty & Calibration
Does uncertainty reliably predict failure?
How miscalibrated are planner outputs under distribution shift?


### Methodology

#### 1. Leave-One-City-Out Training

Train MLPlanner on:
* Boston → test on Boston/Singapore/Pittsburgh
* Singapore → test on Boston/Pittsburgh
* Pittsburgh → test on Boston/Singapore
* Using nuplan_train and nuplan_simulate.

#### 2. Metrics

* Safety: collisions, violations
* Progress: route completion, distance
* Comfort: jerk, acceleration
* Compliance: speed adherence, drivable area
* Uncertainty: MC Dropout variance
* Calibration: ECE, Brier Score, reliability diagrams

#### 3. Scenario-Level Failure Taxonomy

Automatic categorization into:
* unprotected turns
* occluded crossings
* roundabouts
* dense multi-agent scenes
* left-hand vs right-hand failures

Each failure contains:
* scenario tag
* GT vs predicted trajectory
* uncertainty
* violation type
* replay snapshot

#### 4. Visualization Pipeline

* scenario failure heatmaps
* cross-city confusion matrices
* trajectory deviation plots
* reliability diagrams
* scenario replays

#### 5.Current Implementation Status (Jan 2026)

This repository currently implements the offline diagnostic layer of the proposed research agenda, focusing on planner behavior inspection and uncertainty visualization.

Implemented components include:

* Offline trajectory dataset extraction from nuPlan mini logs

* Learning-based planners:

**baseline ego-only planner**

** agent-aware planner

** uncertainty-aware planner (MC Dropout)

* Planner rollout generation in ego-centric coordinates

* Static and animated visualizations comparing:

** ego past

** ground-truth future

** predicted future

** predictive uncertainty

* Interactive Streamlit dashboard for:

** timestep inspection

** uncertainty evolution

** qualitative failure analysis

These tools form the foundation for analyzing planner intent, confidence, and failure modes prior to closed-loop simulation.







