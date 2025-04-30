# From *lead* to *win*

This project analyzes how customer leads move through photovoltaic (PV) sales funnel — from first touch to final contract.
By tracking cohorts across marketing channels and funnel steps, we uncover patterns in conversion rates, sales cycle duration, and lead drop-off.
The goal is to improve sales efficiency by identifying high-performing channels, shortening time-to-conversion, and recognizing inactive leads early.

## Data Overview - Cohort_analysis_dataset


### `lead_id`  
A unique identifier for each lead.  
Used to track the same person across funnel steps.


### `case_opened_date`  
The date when a particular funnel step (e.g., Sales Call 1) started.  
Each step has its own opening date.



### `case_closed_successful_date`  
The date when the funnel step was successfully completed.  
If missing (`NaT`), the step is still open or not successful.



### `sales_funnel_steps`  
Describes which step of the funnel the row refers to (e.g., Sales Call 1, PV System Sold).  
One row = one step.



### `lead_created_date`  
The date when the lead was first generated in the system.  
This is the origin point for all funnel tracking.



### `marketing_channel`  
Indicates the source through which the lead was acquired (e.g., Channel A, Channel B).  
Useful for performance analysis.



### `sales_call_1_flag`  
Binary flag (`1` or `0`) indicating whether the lead reached the Sales Call 1 step.  
Based on presence in the funnel data.



### `sales_call_2_flag`  
Binary flag showing whether the lead progressed to Sales Call 2.  
Helps in tracking funnel progression.



### `pv_system_sold_flag`  
Binary flag indicating whether the lead completed the funnel and resulted in a PV system sale.  
Final conversion signal.



### `cohort_month`  
Month when the lead was created, formatted as `YYYY-MM`.  
Used for cohort analysis and grouping leads by signup time.



### `step_month`  
Month when the funnel step in this row began, based on `case_opened_date`.  
Helps visualize funnel progression over time.



### `days_to_convert`  
Number of days between `lead_created_date` and `case_closed_successful_date`.  
Measures how long it took to complete the step.




## Key insights

[Link](https://docs.google.com/presentation/d/e/2PACX-1vQTdbYOX2Oh7YC57AOtuTOBW2BhIsW4p-7r7SeTqEFIbLQ_UCg-vu7iV948FnJ-5GYSKrqJq6Dj5Jhc/pub?start=false&loop=false&delayms=3000)

## Recommendation


[Link](https://docs.google.com/presentation/d/e/2PACX-1vQTdbYOX2Oh7YC57AOtuTOBW2BhIsW4p-7r7SeTqEFIbLQ_UCg-vu7iV948FnJ-5GYSKrqJq6Dj5Jhc/pub?start=false&loop=false&delayms=3000)