# Data Preprocessing Pro

**Data Preprocessing Pro** is a modular, interactive Streamlit web application that provides a comprehensive suite of tools for preprocessing structured data. It enables users to upload a dataset, perform a variety of preprocessing techniques, and preview the results in real-time — all through an intuitive UI.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [License](#license)
- [Author](#author)

## Features

- **Data Upload**: Upload CSV files for preprocessing.
- **Missing Value Handling**:
  - Drop missing rows or columns
  - Impute using mean, median, or mode
- **Encoding**:
  - Label encoding
  - One-Hot encoding
  - Ordinal encoding
  - Target encoding
- **Scaling**:
  - StandardScaler
  - MinMaxScaler
  - RobustScaler
- **Outlier Detection**:
  - IQR method
  - Z-score
  - Isolation Forest
  - Local Outlier Factor
- **Feature Selection**:
  - Variance Threshold
  - SelectKBest
  - Correlation-based filtering
- **Live Preview**: Inspect the data before and after transformation
- **Export**: Download the processed dataset

## Installation

### Prerequisites

- Python 3.7 or higher

### Clone the Repository

```bash
git clone https://github.com/BettyJk/data-preprocessing-pro.git
cd data-preprocessing-pro
