# Dog Clothing & Harness Size Recommender

Golden Retriever & Labrador Retriever — sized against the real Supertails chart.

## 1. What this project does

Upload a dog photo, enter age/weight/sex (height, and direct
neck/chest/back measurements are optional). The app predicts the breed
from the photo, uses REAL measurements if you gave them (best), or
estimates them from breed + age + weight if you didn't, then matches
against the real Supertails size chart and returns a size + a 0-100 fit
score + a plain-English explanation.

## 2. Folder structure

```
dog-size-recommender/
├── README.md
├── requirements.txt
├── app.py                          # Streamlit UI, orchestrates everything
├── data/
│   ├── breed_growth_reference.csv  # TEMPLATE -- needs real growth-chart data pasted in
│   ├── breed_adult_reference.csv   # weight/height filled from AKC; neck/chest/back blank
│   ├── brand_size_chart.csv        # REAL Supertails data (clothing + harness)
│   └── sample_inputs.csv           # example rows for testing
├── src/
│   ├── __init__.py
│   ├── data_loader.py              # safe CSV loading + column validation
│   ├── utils.py                    # unit conversion + input validation
│   ├── predict_breed.py            # inference (placeholder until trained)
│   ├── measurement_estimator.py    # estimates neck/chest/back if not given
│   └── size_recommender.py         # matches chart, computes 0-100 fit score
├── training/
│   └── train_breed_classifier.py   # trains the ResNet18 breed classifier
└── models/
    └── breed_classifier.pth        # created by training
```

## 3. Data sources -- what's real vs. placeholder

- **`breed_adult_reference.csv`** weight/height columns: REAL, from the
  official AKC breed standards (converted lbs→kg, in→cm). Neck/chest/back
  columns are blank -- AKC standards don't specify girth in cm, so don't
  invent them. Fill from real measured dogs if you get the chance.
- **`breed_growth_reference.csv`**: TEMPLATE ONLY, all weight values blank.
  Paste in real numbers from Pawlicy's Golden Retriever / Labrador growth
  charts, or WALTHAM puppy growth charts. Don't fill from memory.
- **`brand_size_chart.csv`**: REAL data from the Supertails size chart
  screenshot you provided. Branded generically as "Supertails," not
  confirmed Pets Way-specific -- worth double-checking against the actual
  product page if precision matters. Harness rows currently mirror
  clothing rows (per your decision) until you get a harness-specific chart.

## 4. IMPORTANT: quick mode currently can't run

Because `breed_adult_reference.csv` has no neck/chest/back data yet, and
`brand_size_chart.csv` has no `breed_guideline` values that name a specific
breed, `measurement_estimator.py` has nothing to estimate FROM -- it will
correctly return "insufficient data" rather than guess (this was a real bug
I caught and fixed: an earlier version fell back to the largest chart size
as a proxy for "adult," which produced nonsense estimates like a 90cm chest
for a 28kg Labrador).

**To make quick mode work, you need ONE of:**
- Real measured (breed, sex, weight, height) → (neck, chest, back) data for
  a handful of adult dogs, to fill `breed_adult_reference.csv`'s
  neck/chest/back columns, OR
- A breed-specific brand chart (fill `breed_guideline` column in
  `brand_size_chart.csv` with e.g. "Golden Retriever, Labrador").

Until then, the app works in **precise mode only** (user provides real
neck/chest/back measurements directly) -- which is also the most accurate
path anyway.

## 5. Fit score logic (0-100)

Per measurement (chest/back/neck), score depends on where the value sits
within that size's chart range:
- Middle 60% of the range → 85-100 (high fit)
- Outer 20% near either boundary → 60-85 (medium fit)
- Just outside the range (≤5%) → 45 (low but possible)
- Outside by 5-10% → 20 (poor fit)
- Outside by >10% → 0 (reject)

Per-size score = weighted average of available measurements. Weights:
- **Harness**: chest 60%, neck 25%, back 15%
- **Clothing**: chest 45%, back 35%, neck 20%

Then penalties are subtracted from the best size's raw score:
- -8 points per estimated (vs. measured) value used
- -15 if breed confidence <60%, -5 if <85%
- -5 if the dog is under 12 months (puppy, still growing)

Confidence labels: 85-100 High, 65-84 Medium, <65 Low.

## 6. How to Run the App (Setup & Quickstart)

Follow these steps to set up and run the Dog Size Recommender web application locally.

### Step 1: Navigate to the Project Folder
Open your terminal and navigate to the project root directory:
```bash
cd dog-size-recommender
```

### Step 2: Set Up a Virtual Environment (Recommended)
Creating a virtual environment ensures that the project dependencies do not conflict with your global Python setup.
- **macOS / Linux**:
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```
- **Windows**:
  ```bash
  python -m venv venv
  venv\Scripts\activate
  ```

### Step 3: Install Required Dependencies
Install the necessary packages using `pip`:
```bash
pip install -r requirements.txt
```

### Step 4: Run the Streamlit Web Application
Start the Streamlit development server:
```bash
streamlit run app.py
```
After executing this command, the application will automatically compile and open in your default browser at:
👉 **[http://localhost:8501](http://localhost:8501)**

---

### Optional: Train the Breed Classifier
The application is pre-configured to run with a fallback 50/50 prediction model if no classifier is trained. If you wish to train the ResNet18 model on real dataset:
1. Populate `data/breed_images/train/{golden_retriever,labrador_retriever}/` and `data/breed_images/val/{golden_retriever,labrador_retriever}/` with images.
2. Run the training script:
   ```bash
   python -m training.train_breed_classifier
   ```


## 7. Testing with sample inputs

```bash
python3 -c "
from src.data_loader import load_growth_reference, load_adult_reference, load_brand_chart
from src.size_recommender import recommend_size

growth_ref = load_growth_reference()
adult_ref = load_adult_reference()
brand_chart = load_brand_chart()

user_input = {
    'breed': 'Labrador Retriever', 'sex': 'male', 'age_months': 18, 'weight_kg': 28,
    'product_type': 'clothing', 'breed_confidence': 0.91,
    'neck_cm': 38, 'chest_cm': 62, 'back_length_cm': 47,
    'measurement_source': {'neck': 'actual', 'chest': 'actual', 'back': 'actual'},
}
print(recommend_size(user_input, brand_chart, growth_ref, adult_ref))
"
```

## 8. Common mistakes to avoid

- Filling `breed_growth_reference.csv` or the blank neck/chest/back columns
  with guessed numbers instead of real cited sources.
- Trusting a quick-mode estimate as if it were a real measurement -- it's
  penalized in the fit score specifically so it never masquerades as high
  confidence.
- Mixing size charts across different Supertails/Pets Way product lines
  without noting which one you used as source.
- Forgetting `train_ds.classes` order can differ from
  `predict_breed.py`'s hardcoded `CLASS_NAMES` -- always check after training.
