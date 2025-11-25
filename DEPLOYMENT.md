Deployment notes — Amazon Review Analyzer

This project includes trained model artifacts that may be stored via Git LFS locally.
Some hosts (Streamlit Community Cloud) do not automatically fetch LFS objects during deploy.

Recommended configuration for reliable deployment

1) Use external hosting for the model (recommended):

   - Upload the model file (`webapp/models/fine_tuned_xgb.pkl`) to a public storage location (S3, GCS, or similar).
   - In your Streamlit app settings, set the environment variable `MODEL_URL` to the public URL for the model.
     Example:
       MODEL_URL=https://my-bucket.s3.amazonaws.com/fine_tuned_xgb.pkl

   - The app will download the model at startup automatically when it is missing from the instance.

2) Alternatively, ensure your host fetches Git LFS objects:

   - If you prefer to keep models in Git LFS, make sure your deployment environment is configured to pull LFS objects.
   - On some hosts you must enable Git LFS or provide credentials. If LFS objects are not present, the app will fail to load the model.

3) Restart the app after setting `MODEL_URL` or changing repository contents.

4) (Optional) For exact preprocessing parity:

   - The app attempts to load the training TF-IDF/vectorizer from these locations:
     - `scripts/xgb_model/tfidf_vectorizer.pkl`
     - `scripts/xgb_model/vectorizer.pkl`
     - `model/tfidf_vectorizer.pkl`

   - If you want identical predictions to training, upload the vectorizer file to the repository (or to the instance filesystem) in one of the paths above.

Troubleshooting

- If the app still shows "Model file not found", verify that:
  - `MODEL_URL` is set and reachable from the deployment environment.
  - The model URL returns the raw `.pkl` binary (no HTML wrappers).

- If predictions look different from expected, ensure the TF-IDF/vectorizer used during training is available to the app (see locations above).

Contact

If you want, I can add automatic integrity checks (SHA256) and retries for the model downloader. Just tell me and I'll implement them.
