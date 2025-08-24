if __name__ == "__main__":
    import os

    import uvicorn

    from inference_core.main_factory import create_application

    app = create_application()

    APP_PORT = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=APP_PORT)
