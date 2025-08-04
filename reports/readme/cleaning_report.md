# Data Cleaning Report

Generated at: 2025-08-03 21:19:40

## Summary

* Starting shape: 1000 rows × 15 columns
* Ending shape: 1000 rows × 15 columns
* Columns dropped: 0
* Rows dropped: 0
* Columns fixed/modified: 8

## Columns Dropped

No columns were dropped.

## Columns Modified

| Column | Modification |
|--------|-------------|
| school_board | string_cleaning |
| coaching_institute | string_cleaning |
| family_income | string_cleaning |
| parent_education | string_cleaning |
| location_type | string_cleaning |
| peer_pressure_level | string_cleaning |
| mental_health_issues | boolean_conversion |
| admission_taken | boolean_conversion |

## Data Type Changes

| Column | Original Type | New Type |
|--------|--------------|----------|
| mental_health_issues | object | bool |
| admission_taken | object | bool |
