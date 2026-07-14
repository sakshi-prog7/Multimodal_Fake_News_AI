import os
import pandas as pd


possible_files = [
    "ocr_dataset.csv",
    "ocr_data.csv",
    "dataset.csv",
    "image_dataset.csv",
    "multimodal_dataset.csv",
]

csv_file = None

for filename in possible_files:
    if os.path.exists(filename):
        csv_file = filename
        break


if csv_file is None:
    print("\nCSV FILE NOT FOUND")
    print("\nAvailable CSV files:")

    for root, dirs, files in os.walk("."):
        for file in files:
            if file.lower().endswith(".csv"):
                print(os.path.join(root, file))

    raise SystemExit


print("\nUSING CSV:", csv_file)

df = pd.read_csv(csv_file)

print("\n" + "=" * 80)
print("OCR DATA INSPECTION")
print("=" * 80)

print("\nColumns:")
print(df.columns.tolist())

print("\nShape:")
print(df.shape)

print("\nFirst 10 rows:")
print(df.head(10).to_string())


if "label" not in df.columns:
    print("\nERROR: label column not found")
    raise SystemExit


text_column = None

for column in [
    "ocr_text",
    "text",
    "news_text",
    "content"
]:
    if column in df.columns:
        text_column = column
        break


if text_column is None:
    print("\nERROR: Text column not found")
    raise SystemExit


print("\nTEXT COLUMN:", text_column)


print("\n" + "=" * 80)
print("LABEL COUNTS")
print("=" * 80)

print(df["label"].value_counts(dropna=False))


print("\n" + "=" * 80)
print("EMPTY TEXT BY LABEL")
print("=" * 80)

df[text_column] = df[text_column].fillna("").astype(str)

df["text_length"] = (
    df[text_column]
    .str.strip()
    .str.len()
)

for label in sorted(df["label"].unique()):

    subset = df[df["label"] == label]

    empty_count = (
        subset["text_length"] == 0
    ).sum()

    short_count = (
        subset["text_length"] < 10
    ).sum()

    print(f"\nLABEL {label}")

    print("Total       :", len(subset))
    print("Empty Text  :", empty_count)
    print("Text < 10   :", short_count)

    print("\nText Length Stats:")

    print(
        subset["text_length"].describe()
    )


print("\n" + "=" * 80)
print("DUPLICATE TEXT ANALYSIS")
print("=" * 80)

duplicate_count = df.duplicated(
    subset=[text_column],
    keep=False
).sum()

print(
    "Duplicate Text Rows :",
    duplicate_count
)


print("\nTOP DUPLICATE TEXTS:")

duplicates = (
    df.groupby(
        [text_column, "label"]
    )
    .size()
    .reset_index(name="count")
    .sort_values(
        "count",
        ascending=False
    )
)

print(
    duplicates
    .head(20)
    .to_string(index=False)
)


print("\n" + "=" * 80)
print("SAMPLE TEXT BY LABEL")
print("=" * 80)

for label in sorted(df["label"].unique()):

    print(f"\n\nLABEL {label}")
    print("-" * 80)

    samples = df[
        df["label"] == label
    ].head(10)

    for index, row in samples.iterrows():

        print(
            f"\nROW {index}"
        )

        print(
            "Length:",
            row["text_length"]
        )

        print(
            "Text:",
            repr(
                row[text_column][:300]
            )
        )


print("\n" + "=" * 80)
print("INSPECTION COMPLETE")
print("=" * 80)