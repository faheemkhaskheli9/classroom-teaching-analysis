# Architecture Notes: Classroom Teaching Method Analysis

## Pipeline

```text
Classroom Video -> Person/Teacher Detection -> Pose/Action Features -> Action Recognition -> Temporal Analysis -> Teaching-Method Statistics
```

## Components

- Teacher detection in classroom video
- Pose/action feature extraction
- Action recognition (sitting, talking, walking, writing, presenting, pointing)
- Temporal analysis across a lesson
- Teaching-method statistics summary

## Design Notes

- Keep provider/model choices swappable behind interfaces (see `multi-llm-router`
  and similar projects in this portfolio for the general pattern).
- Prefer configuration-driven pipelines (YAML/JSON in `configs/`) over hardcoded
  parameters so experiments are reproducible.
