if __name__ == "__main__":
    import os

    import uvicorn

    from inference_core.core.env import load_project_dotenv
    from inference_core.main_factory import create_application

    load_project_dotenv()

    app = create_application()

    app_port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=app_port)
