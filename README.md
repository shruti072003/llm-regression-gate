# eval-history

Data branch: one line per nightly eval run of the production prompt, appended by
`.github/workflows/nightly.yml` on `main` and read by `gate drift`. Kept off `main` so the
protected branch doesn't need bot pushes and its history stays free of daily data commits.
