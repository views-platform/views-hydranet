
Offline evaluation refers to the process of assessing model performance using historical data prior to any deployment or live operation. It plays a critical role in the model development lifecycle by enabling rigorous experimentation, benchmarking, and validation under controlled and reproducible conditions.

This evaluation framework ensures that our conflict forecasts are tested fairly and meaningfully, reflecting how they will be used in real-world decision-making. By assessing model behavior across different time horizons, regions, and types of violence, we capture both technical performance and operational relevance.

While the framework continues to evolve, it provides a consistent foundation for tracking progress, comparing model variants, and maintaining transparency. Ultimately, offline evaluation is about more than predictive accuracy -- it is about building tools that policymakers can trust when the stakes are highest.


\subsection{Overview and Objectives}

In the VIEWS pipeline, offline evaluation occurs during the R\&D phase -- before models are deployed as shadow or production systems. The approach supports rolling, time-aware train/test splits within each data partition (Calibration, Validation, Forecast), simulating a realistic sequence of model development, tuning, and retrospective forecasting. This design departs somewhat from conventional static partitioning to better accommodate the non-stationarity of conflict data.

Each data partition corresponds to a different stage in historical time, and supports distinct modeling goals: Calibration for initial development, Validation for model selection and robustness checks, and Forecasting for system-level benchmarking. Within each, rolling training and forecasting horizons are constructed using sequences of 36 months of input data and up to 48 months of predictive output. This rolling framework supports step-wise evaluation as the default, while allowing for additional styles such as time-series-wise and month-wise evaluation.

The key objective of offline evaluation is to simulate how the system would have performed if it had been deployed in the past, using fixed hold-out partitions. Retrospective testing like this enables us to assess model behavior on known data and identify both pointwise accuracy (e.g., forecast error) and broader behavioral patterns, such as persistent underprediction in certain regions or instability during volatile periods. These diagnostics are critical for refining model specifications before operational deployment.

Offline evaluation supports several key functions:
\begin{itemize}
\item Guiding model selection and hyperparameter tuning (e.g., comparing competing model architectures),
\item Establishing benchmarks across historical baselines and model generations,
\item Stress-testing robustness under rare events or edge-case scenarios,
\item Enabling reproducible comparisons for internal review and academic dissemination.
\end{itemize}

As such, offline evaluation serves as both a quality control gate before investing further in model deployment and as the primary means of documenting performance for external audiences, ex ante deployment. The partitioning scheme, adapted from modern multivariate time-series approaches (e.g., Darts), allows for alignment with broader ML standards while preserving VIEWS-specific needs. 

%A translation table to common ML and time-series terminology is maintained to ensure interpretability across audiences.

\subsection{Data Partitioning Strategy}

The VIEWS offline evaluation framework is structured around three temporal partitions -- Calibration,  Validation, and  Forecast -- each designed to reflect a different phase of the model lifecycle. Crucially, each partition contains its own train/test split, enabling us to simulate development, benchmarking, and deployment under realistic historical constraints.

\paragraph{Calibration Partition:} 
This partition supports exploratory model development using older historical data. It is used to train initial models, conduct feature exploration, and make early architecture decisions. Because development is iterative and intensive, models are often evaluated repeatedly on the calibration test set -- leading to a risk of overfitting. As a result, this partition provides insight into model potential but not true out-of-sample performance.

\paragraph{Validation Partition:}
To safeguard against overfitting on the calibration set, the validation partition serves as a clean test environment. Once a model specification is considered finalized, it is retrained on the validation training set and evaluated on the validation test set -- data it has not been exposed to during development. This partition is central for model selection, robustness testing, and academic dissemination, as it provides a fair benchmark of performance on unseen data.

\paragraph{Forecasting Partition:}
This partition is for live deployment. Forecasts are generated using only data that would have been available at the time of prediction, ensuring no data leakage. Unlike the other partitions, there are no observed outcomes to test against yet. As such, this partition represents operational output rather than a test of past performance.

The calibration and validation partitions are updated annually, typically in July, to align with system retraining and UCDP annual updates \citep{UCDP_2017}, while the forecasting partition is updated monthly in alignment with the UCDP candidate dataset \cite{hegre2020introducing} to reflect ongoing live predictions. 

\input{tables/data_partitions}

\subsection{The Predictive Parallelogram}

The Calibration and Validation partitions are each defined over a distinct historical period, and evaluation is performed using a sliding-window approach. Specifically, Models are trained on a 36-month rolling input window and evaluated across a 48-month forward prediction window. **[NEEDS REVIEW: This '48-month' window contradicts the '36-month' forecast sequence length defined elsewhere in this document and in ADR-002. This should be clarified and made consistent.]** This setup enables 12 sub-evaluations per test window: after each forecast, the input window is rolled forward one month, and the forecasting procedure is repeated. Stacking these overlapping forecast runs forms a predictive parallelogram in calendar time -- a structure that supports robust temporal evaluation and mimics real-time deployment cadence.

\begin{figure}
    \centering
    \includegraphics[width=1\linewidth]{figures/approach.png}

    \begin{picture}(0,0)\put(0,100){\makebox(0,0){\rotatebox{45}{\textcolor{gray!50}{\fontsize{100}{100}\selectfont \textbf{ILLUSTRATION BY MIHAI}}}}}\end{picture} % MIHAI WATERMARK
    
    \caption{Evaluation strategy -- needs review}
    \label{fig:evaluation strategy}
\end{figure}

The general evaluation strategy is illustrated in Figure \ref{fig:evaluation strategy}. It involves training one model on a time series that goes up to the training horizon $H_0$. This sequence is then used to predict a number of sequences (time-series). The first such sequence goes from $H_{0+1}$ to $H_{0+36}$, thus containing 36 forecasted values -- i.e. 36 months. The next one goes from $H_{0+2}$ to $H_{0+37}$. This is repeated until we reach a constant stop-point $k$ such that the last sequence forecasted is $H_{0+k+1}$ to $H_{0+k+36}$. 

This design supports a diverse range of modeling paradigms (e.g., autoregressive, direct multi-step, sequence-to-sequence), promotes fairness in benchmarking, and enables flexibility for evolving ensemble strategies.

To analyze forecast performance across time and space, the VIEWS framework applies three complementary evaluation schemes. These are detailed in the following subsection.

\subsection{Time-series-wise Evaluation} \label{sec:time-series-wise}

In VIEWS, time-series-wise evaluation assesses model performance across the entire 36-month forecast sequences. Each forecast is aligned with observed outcomes, and a single aggregate score (e.g., RMSE or CRPS) is computed for the full sequence. This approach provides a high-level summary of model accuracy and is commonly used in libraries like \texttt{Darts} and \texttt{skforecast}.

Unlike step-wise evaluation, which groups predictions by forecast lead time, time-series-wise evaluation groups them by prediction sequence. In the predictive parallelogram, this corresponds to evaluating each row: a 36-month forecast issued from a given start date for a specific spatial unit. In the VIEWS setup, the evaluation window spans 48 months, and the input window is rolled forward by one month after each forecast. This results in 12 overlapping forecast sequences per evaluation window, each of which yields one metric. This structure is illustrated in Figure~\ref{fig:ts}.

\begin{figure}
    \centering
    \includegraphics[width=1\linewidth]{figures/ts.png}
    \begin{picture}(0,0)\put(0,100){\makebox(0,0){\rotatebox{45}{\textcolor{gray!50}{\fontsize{100}{100}\selectfont \textbf{ILLUSTRATION BY MIHAI}}}}}\end{picture}
    \caption{Time-series-wise evaluation. Each vertical slice corresponds to a 36-month forecast sequence, evaluated as a single unit.}
    \label{fig:ts}
\end{figure}

While this method reflects typical practices in machine learning libraries, it can obscure differences between short- and long-term model performance. Because errors are averaged across all forecast steps, poor long-horizon predictions may be hidden by strong near-term performance. In contrast, VIEWS emphasizes step-wise evaluation, which computes a distinct score for each forecast step (e.g., 1 month ahead, 36 months ahead). This allows a more granular assessment of model behavior -- crucial for applications like conflict forecasting, where short-term reactivity and long-term structural foresight often require different modeling strategies.

Additionally, time-series-wise evaluation tends to favor conservative models that track long-term trends. Because it aggregates error across entire 36-month forecast sequences, this approach may reward models that fit overall trajectories while overlooking short-term volatility or rare but sharp disruptions -- such as sudden conflict escalation. As a result, models may achieve high average performance while systematically missing critical inflection points that step-wise evaluation would expose.

Despite this limitation, time-series-wise evaluation enables important analytical techniques that require continuous prediction sequences\footnote{Such as Granger causality analysis or Sinkhorn distance comparisons, as these methods rely on comparing full trajectories or distributional structures and are only valid when forecasts are evaluated as coherent sequences.}. It also supports flexible spatial aggregation, whether at the country level or at finer grid-cell resolution.

While not the primary evaluation method in VIEWS' operational workflows, time-series-wise evaluation remains a standard in academic machine learning toolkits. It offers a complementary perspective to step-wise diagnostics -- especially when evaluating structural realism, causal patterns, or long-range fit.


\subsection{Step-wise Evaluation}

Step-wise evaluation is the most emphasized and commonly referenced evaluation strategy in the VIEWS system. While all three evaluation schemes are used concurrently, step-wise analysis is typically the first examined and most central to model interpretation and benchmarking workflows.

This approach is designed to assess how predictive skill varies with lead time -- i.e., how well models forecast events at different distances into the future. Each forecast step, from 1 to 36 months ahead, is evaluated independently. For each step $s$, all predictions made with a lead time of $s$ months -- across all forecast issuance dates and spatial units (sub-national grids or countries) -- are collected. These predictions are then aligned with their corresponding ground truth observations and scored using appropriate evaluation metrics. The result is a \textbf{set of 36 step-specific performance scores per model}, one for each forecast horizon.

This structure is illustrated in Figure~\ref{fig:step}, where each diagonal in the predictive parallelogram corresponds to a single forecast step. These diagonals represent rows of the forecast matrix: each connects predictions made with the same lead time across multiple forecast issuance dates and spatial units.

\begin{figure}
\centering
\includegraphics[width=1\linewidth]{figures/steps.png}
\begin{picture}(0,0)\put(0,100){\makebox(0,0){\rotatebox{45}{\textcolor{gray!50}{\fontsize{100}{100}\selectfont \textbf{ILLUSTRATION BY MIHAI}}}}}\end{picture}
\caption{Step-wise evaluation. Each diagonal corresponds to a forecast step (1 to 36 months ahead), linking predictions made with a fixed lead time across all forecast runs.}
\label{fig:step}
\end{figure}

Step-wise evaluation is particularly valuable in conflict forecasting, where model performance often varies substantially across short and long horizons. Some models respond to immediate signals -- excelling at predicting events just 1--2 months ahead -- while others better capture slower structural dynamics, such as escalation patterns, that manifest over 18 to 36 months. A step-specific breakdown reveals such differences and helps avoid misleading aggregate scores -- for example, a model that performs poorly beyond month 12 might still appear strong when evaluated using time-averaged metrics.

These results are also critical for ensemble modeling. Forecast combinations can be weighted by step, assigning more importance to models that perform better at specific horizons. For instance, a nowcasting model might dominate short lead times, while a structurally informed model provides superior long-range accuracy.

A common point of confusion in earlier documentation is the distinction between a \textbf{step} and a \textbf{stride}. The following table summarizes the difference:

\input{tables/step_v_stride}

As such, \textit{Step} defines what the model is trying to predict, while \textit{Stride} defines how often new training sequences are generated. The two concepts are distinct and must not be conflated -- especially when interpreting evaluation results.

VIEWS primarily employs an \textit{expanding-window evaluation strategy}, where models are retrained periodically (typically every 12 months) using all available data up to that point. However, the step-wise framework itself is agnostic to whether an expanding or rolling window is used. What matters is that all predictions are grouped by lead time and aligned across forecast issuance dates, preserving the integrity of the step-wise breakdown.

Step-wise evaluation is not typically supported in standard time-series libraries like \texttt{Darts} or \texttt{Prophet}, which focus on time-series-wise averaging. VIEWS emphasizes horizon-specific performance because operational decisions often depend on understanding whether models perform differently at short, medium, or long-range horizons -- a particularly important consideration in non-stationary settings like political violence forecasting.



\subsection{Month-wise Evaluation}

Month-wise evaluation isolates model performance for a specific calendar month in the test set of a given partition (calibration or validation) -- such as January 2018 or February 2022. Rather than aggregating over lead times or full sequences, it focuses on a single target month in historical time, evaluating how well models predicted outcomes during that fixed period.

In the predictive parallelogram, this corresponds to selecting a column or horizontal slice: all predictions that target the same month -- regardless of when the forecast was issued -- are collected and scored against observed outcomes. This allows for detailed inspection of temporal anomalies or periods of heightened interest. This structure is illustrated in Figure~\ref{fig:month}.

\begin{figure}
    \centering
    \includegraphics[width=1\linewidth]{figures/months.png}
    \begin{picture}(0,0)\put(0,100){\makebox(0,0){\rotatebox{45}{\textcolor{gray!50}{\fontsize{100}{100}\selectfont \textbf{ILLUSTRATION BY MIHAI}}}}}\end{picture}
    \caption{Month-wise evaluation. Each horizontal slice corresponds to predictions for a specific calendar month in the test set (e.g., January 2018). Illustration by Mihai.}
    \label{fig:month}
\end{figure}

This approach is particularly useful for understanding model behavior around time-specific disruptions or critical historical events -- such as March 2014 (annexation of Crimea), February 2022 (the Russian invasion of Ukraine), or October 2023 (Israel–Hamas war). Because each test month occurs only once per evaluation run, sample sizes can vary depending on how many forecasts target that month. This unevenness affects metric stability and interpretability, especially for rare-event metrics where small count shifts can have outsized effects.

Because month-wise evaluation focuses on a single target month, multiple forecasts -- issued at different times -- may end up predicting that month using overlapping or nearly identical training data. This creates a risk of unintended correlation: the model’s performance on that month may reflect shared training context rather than independent generalization. As a result, apparent consistency in performance may be inflated. To reduce this risk, partitioning and training windows must be structured to limit overlap when high independence is required.

Month-wise evaluation is well-suited for inspecting how models behave during specific historical periods of interest. Isolating performance on a single month in a test set enables focused analysis of model responsiveness to sharp disruptions, seasonal variation, or rare events. While its role within formal evaluation pipelines is still being shaped, it provides unique diagnostic insight, particularly in forecasting environments where the timing of events carries strategic importance.

\subsection{Next Steps}
%Internal notes: This section will eventually summarize all remaining action items, gaps, and planned improvements in the evaluation pipeline. It should serve as a running list of what still needs to be implemented (e.g., metric automation, infrastructure, online eval deployment, documentation, calibration procedures). For now, just keep it as a placeholder to collect actionable to-dos and signal ongoing development.    


To strengthen the evaluation framework and enhance its operational utility, the following key improvements will be prioritized:

\begin{itemize}
    \item \textbf{Metric Implementation} \\
    Expand the \textit{}{views-evaluation} package to include all planned metrics beyond the current focus on RMSLE, CRPS, and AP. This will enable a comprehensive assessment of both calibration and sharpness across all forecast horizons.
    
    \item \textbf{Baseline Model Deployment} \\
    The three baseline models will be implemented. These will serve as a reference for model comparison, enabling clearer performance interpretation and more robust validation of improvements.

    \item \textbf{Online Evaluation System} \\
    Online evaluation can continuously validates predictions against UCDP candidate data for out-of-sample forecasts
\end{itemize}

These enhancements will help improve the end-to-end views pipeline that incorporates rigorous metric assessment.




