# NFL Championship DNA: Quantifying Playoff Success

## Project Overview
This project applies advanced machine learning techniques to NFL play-by-play data to answer a fundamental question: "What traits are most critical for winning in the playoffs?" By building a "Micro-Model" that understands the physics of individual plays, we generate proprietary metrics (like Success Over Expected) to evaluate teams independent of their schedule difficulty. We then aggregate these metrics to identify the specific team profiles-or "Championship DNA"-that correlate with Super Bowl runs.

## Problem Statement
Traditional NFL analysis often relies on raw volume stats (Total Yards) or efficiency metrics (EPA) that are biased by opponent quality. 
* Does an elite run game matter more than an elite secondary?
* Is "Clutch Factor" real, or just noise?

This project seeks to isolate Situational Execution from Scheme/Luck to determine what actually translates to postseason success in the Modern Era (2016-Present).

## Methodology

### Phase 1: The Micro-Engine (Play Predictor)
We trained an XGBoost Classifier on 13,000+ plays to predict the probability of success (Positive EPA) for any given play before the ball is snapped.
* Input: Pre-snap context (Down, Distance, Score, Alignment, Motion, Box Count).
* Output: Expected Success Rate (xSuccess).
* Result: A baseline model (~64% accuracy) that establishes the "difficulty" of every situation.

### Phase 2: The Macro-Analysis (Team Profiling)
Using the Micro-Engine, we calculate Success Over Expected (SOE) for every team.
* Execution metric: How often does a team succeed when the math says they should fail?
* Clustering: We use K-Means Clustering to group teams into archetypes (e.g., "Glass Cannons", "Defensive Juggernauts").
* Inference: We correlate these advanced traits with actual playoff results (Rounds Advanced) to rank the Top 10 predictors of a Super Bowl Champion.

## Project Structure

| File | Description |
| :--- | :--- |
| 01_feature_engineering_model_build.ipynb | Data cleaning, feature engineering (70+ signals), and training the XGBoost "Micro-Model". |
| 02_playoff_success_analysis.ipynb | Loading the model, generating "Success Over Expected" scores, and running the correlation/clustering analysis. |
| data/epa_success_model_v1.pkl | The trained, serialized XGBoost model artifact. |
| data/enriched_plays_v1.parquet | The processed dataset containing engineered features for analysis. |

## Getting Started

1. Install Dependencies
   ```bash
   pip install -r requirements.txt
   ```

2. Run the Analysis
   * Open 01_feature_engineering_model_build.ipynb to see the model training process.
   * Open 02_playoff_success_analysis.ipynb to see the final insights and visualizations.

## Key Findings
*(To be populated upon completion of Phase 2)*
* Top Predictor: [Pending Analysis]
* Most Overrated Trait: [Pending Analysis]

---
*Author: Bruce Gavins*
