# Real-Time-Adaptive-Cross-Media-Recommendation-System
This project aims to build a cross media recommendation system which adapts in real time according to user's taste of media selection , using semantic embeddings and graph based collaborative learning to understand the user's taste and provide precise recommendations in an unified platform (music + movies + songs)
Takes the description from the data set ---> Sentence Bert ----> Match

## Backend API

Start the recommendation API:

```powershell
.\.venv\Scripts\Activate.ps1
uvicorn backend.app:app --reload
```

Example request:

```powershell
Invoke-RestMethod -Method Post `
  -Uri http://127.0.0.1:8000/recommend `
  -ContentType "application/json" `
  -Body '{"query":"space adventure with aliens","top_k":5,"content_type":"movie"}'
```
