if __name__ == "__main__":
    import os

    import uvicorn

    from app.main import app

    APP_PORT = int(os.getenv("PORT", "8080"))
    uvicorn.run(app, host="0.0.0.0", port=APP_PORT)
