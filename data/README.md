# Dataset

This project trains on the **[Spam Assassin Email Classification Dataset](https://www.kaggle.com/datasets/ganiyuolalekan/spam-assassin-email-classification-dataset)**
(Kaggle, uploaded by ganiyuolalekan). The raw CSV is not committed to this
repo — download it yourself and drop it here.

## Get the data

1. Download the dataset from the Kaggle link above (sign-in required).
2. Unzip it and place the CSV in this folder as:
   ```
   data/spam_assassin.csv
   ```
3. The training notebook auto-detects the text and label columns (common
   variants: `text`/`body`/`Body` for the email content, `target`/`label`/`spam`
   for the 0=ham / 1=spam flag), so it doesn't matter exactly which column
   names your download uses.

## What this dataset actually contains

Spam vs. ham only — there is no "priority" label. The training notebook
(`notebooks/email_priority_training.ipynb`) derives a `high` / `medium` / `low`
priority label with a rule-based pass over the email text (Section 3 of the
notebook): spam → `low`, ham with urgent/time-sensitive language → `high`,
ham with routine-work language → `medium`, everything else → `low`.

That's a reasonable v1 shortcut, not a validated ground truth. If you want a
stronger model, hand-label a sample against the weak labels and check they
agree before trusting downstream numbers.

## License

Refer to the dataset's Kaggle page for its license and usage terms before
redistributing it elsewhere.
