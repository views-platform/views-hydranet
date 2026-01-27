# Core Concepts in VIEWS Evaluation

This document explains the core concepts behind the `views-evaluation` framework, clarifying how data is organized and how model performance is measured.

## 1. Data Organization: Partitions and Sets

The framework uses a two-level data separation strategy to ensure robust and realistic model assessment.

### Level 1: Partitions (The "When")

Partitions are large, distinct, non-overlapping blocks of historical time. They separate the model lifecycle into distinct stages.

-   **Calibration Partition:** The oldest block of data, used for initial research and development, feature engineering, and experimental training.
-   **Validation Partition:** A more recent block of "clean" historical data the model has not seen during development. It is used for the final, fair, out-of-sample benchmarking of a finalized model. This is where performance metrics for academic papers are generated.
-   **Forecasting Partition:** The most recent data, used to generate live, operational forecasts. It has no ground-truth outcomes to test against yet.

**Analogy:** Think of Partitions as different books in a history series (e.g., *Vol. 1: The Early Years*, *Vol. 2: The Middle Era*).

### Level 2: Sets (The "How")

Within the Calibration and Validation partitions, data is further divided into `train` and `test` sets.

-   **Train Set:** The portion of a partition's data used to train a model.
-   **Test Set:** The remaining portion of that partition's data used to evaluate the model's performance.

**Analogy:** Within each book (Partition), you use some chapters to study (the `train set`) and the remaining chapters for a quiz (the `test set`).

---

## 2. The Predictive Parallelogram

The standard offline evaluation process uses a rolling-origin strategy. A model is trained and used to predict a 36-month sequence. The training window is then rolled forward one month, and the process repeats. When stacked, these 12 overlapping forecast sequences form a **predictive parallelogram**.

This parallelogram is the fundamental data structure that is analyzed by the three evaluation schemas.

## 3. The Three Evaluation Schemas

The `EvaluationManager` assesses the predictive parallelogram by "slicing" it in three different ways. Each schema groups the data differently to answer a unique question about model performance.

### Schema 1: Time-series-wise Evaluation

-   **Grouping Method:** Groups predictions by **forecast run**. Each of the 12 forecast sequences is evaluated as a single, complete unit. This is a "vertical slice" of the parallelogram.
-   **Question Answered:** "How good was the model's entire 36-month forecast, on average, when it was issued from a specific start time?"
-   **Analogy:** Getting a single, overall grade for an entire essay.

### Schema 2: Step-wise Evaluation

-   **Grouping Method:** Groups predictions by **forecast horizon** (or lead time). All "1-month-ahead" predictions are grouped, all "2-months-ahead" are grouped, and so on. This corresponds to the "diagonals" of the parallelogram.
-   **Question Answered:** "How does the model's accuracy change as it predicts further into the future?" This is the most critical evaluation schema in the VIEWS framework.
-   **Analogy:** Grading the quality of all the *introduction paragraphs* from a batch of essays, then all the *body paragraphs*, then all the *conclusions* separately.

### Schema 3: Month-wise Evaluation

-   **Grouping Method:** Groups all predictions that target the **same calendar month**, regardless of when the forecast was issued. This is a "horizontal slice" of the parallelogram.
-   **Question Answered:** "How well did the system predict the events of March 2022, using all forecasts that targeted that specific month?"
-   **Analogy:** Grading every student's answer to "Question #5" on a test.

---

### Summary Table

| Evaluation Schema   | Groups Predictions By... | Question It Answers                                      | Analogy                                   |
| ------------------- | ------------------------ | -------------------------------------------------------- | ----------------------------------------- |
| **Time-series-wise**| Forecast Run             | "How good was an entire 36-month forecast?"              | Grading a whole essay.                    |
| **Step-wise**       | Forecast Horizon (Step)  | "How good is the model at predicting 6 months out?"      | Grading all introductions separately.     |
| **Month-wise**      | Target Calendar Month    | "How well did we predict the events of a specific month?" | Grading all answers to one test question. |
