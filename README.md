# AEGIS SOC Dashboard and Attack Simulator

## Backend

```powershell
cd server
Copy-Item .env.example .env
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python app.py
```

The API is available at `http://localhost:5050`. Port 5050 avoids the existing AEGIS gateway's port 5000. It imports and executes the existing `src.router.route_traffic()` policy with safe synthetic traffic profiles. Set `PORT` and `CORS_ALLOWED_ORIGINS` in `server/.env` (or your hosting provider) for deployment.

## Frontend

```powershell
cd client
Copy-Item .env.example .env
npm install
npm run dev
```

Set `VITE_API_BASE_URL` in `client/.env` to the deployed Flask URL when hosting the client and server separately.
