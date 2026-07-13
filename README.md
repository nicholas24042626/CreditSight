# CreditSight

CreditSight is a credit risk analysis website built with Node.js, Express, and Python machine learning scripts.

## Folder layout

- `app.js` - main Node.js server
- `public/` - frontend files
- `python/` - preprocessing, training, and prediction scripts
- `models/` - saved trained models and preprocessing pipelines
- `uploads/` - uploaded company datasets
- `outputs/` - generated prediction CSV files
- `data/` - Dataset A for training

## Run steps

1. `pip install -r requirements.txt`
2. `npm install`
3. `python train_models.py --data "path\to\set A corporate_rating.csv"`
4. `node app.js`

## Notes

- The training script expects a rating column such as `Rating`, or you can pass `--target-column <column_name>`.
- The app uses the saved models only, so it does not retrain during uploads.
- Uploaded company files are used for prediction only.
- If your dataset uses slightly different column names, update the aliases in `python/common.py`.
