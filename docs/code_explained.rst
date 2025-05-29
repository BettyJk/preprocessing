Code Structure and Explanation
==============================

This section explains the core code of the app, split into modules for clarity.

Main App File: ``preprocessing.py``
-------------------------

- Initializes Streamlit and sets the page layout
- Handles file upload and target column selection
- Displays sidebar options
- Applies preprocessing steps in user-defined order
- Renders processed data and visualizations

Key Functions
-------------

``handle_missing_data(df)``
^^^^^^^^^^^^^^^^^^^^^^^^^^^
- Offers three strategies: drop rows, SimpleImputer, or KNNImputer
- Supports different imputation logic per column
- Visual UI for selecting method

``encode_categorical(df)``
^^^^^^^^^^^^^^^^^^^^^^^^^^
- Supports One-Hot, Ordinal, Label, and Target encoding
- Automatically detects categorical columns
- Encodes with intuitive dropdown interface

``scale_features(df)``
^^^^^^^^^^^^^^^^^^^^^^
- Offers MinMaxScaler, StandardScaler, and RobustScaler
- Applies transformation to all numeric columns

``handle_outliers(df)``
^^^^^^^^^^^^^^^^^^^^^^^
- Supports IQR filtering, Z-score, Isolation Forest, and LOF
- Threshold settings are fully interactive

``feature_selection(df, target_col)``
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
- Supports Variance Threshold, SelectKBest, and Correlation Filtering
- Helps reduce dimensionality and improve model performance

Streamlit Features
------------------

- Uses ``st.subheader``, ``st.selectbox``, ``st.slider``, ``st.download_button``
- Embeds plots with ``st.pyplot``
- Ready to optimize with ``st.cache``
