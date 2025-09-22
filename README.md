This project builds a machine learning model to estimate the market value of football players. Using historical data such as age, position, physical and technical attributes, potential, and league level, it trains position-specific XGBoost models (goalkeepers, defenders, midfielders, wingers, forwards).

To improve accuracy, new interaction features are introduced—for example, combining age with potential, relating market value to league level, and calculating a score that links body type to playing position. The workflow also applies winsorization to reduce the effect of outliers and uses targeted feature selection to focus on the most relevant variables.

The final model, validated with 5-fold cross-validation, generates predictions on the test dataset and outputs a clean, submission-ready file. In addition, the project highlights low-importance features for each position group, offering insights into how the model makes decisions.
