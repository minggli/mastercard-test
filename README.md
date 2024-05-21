## Technical test submitted by Ming Li

#### Setup
run `poetry install` with Python 3.11 to set up virtual environment and required dependencies to execute the notebook or script (main.py)

#### Data processing
Categorical values in data are mapped to integer values according to a static ascii printable character set. This means should a new category is added to any feature, existing codification will remain applicable.

#### Model fitting
Model selection and hyperparameter tuning were carried out to choose best model configuration.

Even with only top 5 features from permutation importance (without clear sign of information leakage), test set performance is outstanding for almost all candidate models, as shown in confusion matrix and ROC curve.

However, logistic regression indeed requires one-hot encoded feature set or its performance will suffer.

#### Analysis
Analysis using Shapley values indicate local and directional feature contribution at each test example, alongside global feature importance. The global feature importance from Shapley values roughly agrees with the result of permutation importance.

#### Recommendations
one poisonous examples indiciates (as shown in decision plot) that narrow gill size, together with pugent odor may indiciate the mushroom is poisonous. and further analysis can be carried out and establish exactly which types of mushrooms are to be avoided.
