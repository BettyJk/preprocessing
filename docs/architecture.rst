Application Architecture
========================

The architecture of **Data Preprocessing Pro** is modular, readable, and highly extensible.
Each module has a clear responsibility in the pipeline and is developed using Python best practices.

Layered Design Overview
-----------------------

1. **User Interface (Streamlit):**
   - Handles layout, file uploads, sidebar configuration, and user feedback.
   - Uses ``st.file_uploader``, ``st.selectbox``, and ``st.multiselect`` to create an interactive experience.

2. **Data Management (Pandas, NumPy):**
   - Reads files (``.csv``, ``.xlsx``, ``.txt``)
   - Provides efficient DataFrame operations
   - Ensures memory-efficient handling of large datasets

3. **Preprocessing Engine (Scikit-learn):**
   - Performs imputation, encoding, outlier removal, scaling, and feature selection
   - Uses ``Pipeline`` and ``ColumnTransformer`` logic where applicable

4. **Visualization (Seaborn, Matplotlib):**
   - Displays histograms, countplots, and KDE plots
   - Enables visual checks pre- and post-processing

5. **Export System (Joblib, CSV Writer):**
   - Allows downloading the cleaned dataset
   - Enables future integration with ML model export

Module Interaction
------------------

.. code-block:: text

   [User Uploads Data] → [Handle Missing Values] → [Encode Categorical] → [Outlier Detection]
   → [Normalization] → [Feature Selection] → [Download + Visualization]

All components are loosely coupled, enabling future enhancements like AutoML, logging, or dataset versioning.
