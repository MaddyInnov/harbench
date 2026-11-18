# harbench

Honest-protocol HAR model comparison code (subject-disjoint splits, validation-only selection, test evaluated once).

Used by the Kaggle notebook "09a4 pamap2 completion" to finish the PAMAP2 comparison:
gated-deltanet seeds 43,44; tinyhar-grkan, rwkv-ts and inceptionkan seeds 42,43,44.

Layout expected by the driver: this repo's `harbench/` package and `driver_compare.py`
sit together in `/kaggle/working`.
