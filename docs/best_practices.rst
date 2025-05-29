Optimization and Best Practices
===============================

This section highlights practical advice to ensure you get the best from **Data Preprocessing Pro**.

Memory Management
-----------------

- Convert columns to efficient ``dtype`` (e.g., ``category``)
- Drop unused columns early
- Use ``st.cache`` for expensive computations (future feature)

Security Tips
-------------

- Validate all user inputs (especially for future feature uploads)
- Avoid storing user data on disk
- Run inside Docker for sandboxing

Performance Boosts
------------------

- Use ``astype()`` to optimize column memory
- Apply ``SelectKBest`` early to trim feature space
- Keep plots simple when working with large datasets

Pipeline Stability
------------------

- Always follow recommended preprocessing order:
  1. Missing Values
  2. Encoding
  3. Outliers
  4. Scaling
  5. Feature Selection
- Stick to same pipeline logic across train/test/production
