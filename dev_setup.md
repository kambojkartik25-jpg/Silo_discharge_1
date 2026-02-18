# Fullstack Dev Setup (Backend + Frontend)

## 1) Backend environment

From repo root (`C:\Silo_discharge`):

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## 2) Frontend dependencies

```powershell
cd frontend
npm install
cd ..
```

## 3) Frontend API URL

`frontend/.env` is configured as:

```env
VITE_API_URL=http://localhost:8000
```

## 4) Run backend (dev)

```powershell
python -m uvicorn silo_blend.api.app:app --host 127.0.0.1 --port 8000 --reload
```

Backend endpoints:
- `GET http://127.0.0.1:8000/health`
- `POST http://127.0.0.1:8000/simulate`
- `POST http://127.0.0.1:8000/optimize`
- OpenAPI docs: `http://127.0.0.1:8000/docs`

## 5) Run frontend (dev)

```powershell
cd frontend
npm run dev
```

Frontend default URL:
- `http://localhost:5173`

## 6) One-command dev run

From repo root:

```powershell
python run_dev.py
```

or on Windows PowerShell:

```powershell
.\run_dev.ps1
```

These start:
- backend on `http://127.0.0.1:8000`
- frontend dev server on `http://localhost:5173`

## 7) Integration test flow

1. Open frontend at `http://localhost:5173`
2. Click Optimize in UI
3. Confirm backend terminal logs show `POST /optimize ...`
4. Confirm no CORS errors in browser console
5. Confirm optimize response JSON is returned and UI renders results

## 8) Quick API check

```powershell
curl http://127.0.0.1:8000/health
```
