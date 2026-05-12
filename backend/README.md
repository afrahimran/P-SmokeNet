# P-SmokeNet backend

# Backend structure
P-SmokeNet/
          backend/
                  main.py
                  config.py
                  modelling.py
                  schema.py
                  repository.py
                  README.md
                  requirements.txt
                  demo_videos/
                              video10.mp4
                              video23.mp4
                              video27.mp4
                              video55.mp4

# Backend setup
cd backend

python -m venv .venv
source .venv/bin/activate # (Windows: .venv\Scripts\activate)

pip install -r requirements.txt

# Run backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# API endpoints 
- Health check
GET /health

- Model metadata
GET /api/meta

- Experiment summary
GET /api/experiments/w2p0s_l2p0s_s0p2s/abl_w2p0s_l2p0s_s0p2s_fusion/summary

- Timeline prediciton (preloaded videos)
GET /api/inference/timeline/{video_id}

- Upload video for prediction 
POST /api/inference/upload-video

Body (form-data) :
  file = <video file>