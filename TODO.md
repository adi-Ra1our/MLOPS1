# Cancer ML Project TODO

1. CI/CD Automation
- [ ] Set up GitHub Actions (or similar) for automated testing and deployment on every push.

2. Deployment
- [ ] Add a `Dockerfile` to containerize the FastAPI app.
- [ ] Add deployment scripts for cloud, on-prem, or Docker Compose.

3. Monitoring & Logging
- [ ] Add logging to the API for requests and errors.
- [ ] Optional: add monitoring for model drift and API health.

4. Documentation
- [ ] Expand the `README.md` with API usage examples (input/output format).
- [ ] Expand the `README.md` with setup and run instructions.
- [ ] Expand the `README.md` with a pipeline diagram.

5. Advanced MLOps (Optional)
- [ ] Integrate a remote model registry such as an MLflow server, S3, or Azure ML.
- [ ] Add automated retraining or a batch inference pipeline.

6. Data Validation
- [ ] Add explicit schema validation and outlier detection for incoming data.
