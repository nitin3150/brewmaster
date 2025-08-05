# BrewMaster

This is a web application with a React frontend and a Python backend.

## To Run the Application Manually:

### Frontend

1.  Navigate to the `frontend` directory:
    ```bash
    cd frontend
    ```
2.  Install the dependencies:
    ```bash
    npm install
    ```
3.  Start the development server:
    ```bash
    npm run dev
    ```
    The frontend will be available at `http://localhost:3000`.

### Backend

1.  Navigate to the `backend` directory:
    ```bash
    cd backend
    ```
2.  Install the dependencies from `requirements.txt`:
    ```bash
    pip install -r requirements.txt
    ```
3.  Start the backend server:
    ```bash
    uvicorn server:app --reload
    ```
    The backend will be running on `http://localhost:8000`.
