"""Backward-compatible entry point for existing deployments."""

from app import app

if __name__ == "__main__":
    import config

    app.run(host=config.HOST, port=config.PORT, debug=config.DEBUG)
