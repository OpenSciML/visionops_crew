import os

from dotenv import load_dotenv

load_dotenv(".env")

import fiftyone as fo


def main() -> None:
    """Launch the FiftyOne App local server and wait for it to be closed."""
    session = fo.launch_app(
        address=os.getenv("FIFTYONE_APP_ADDRESS", "0.0.0.0"),
        port=int(os.getenv("FIFTYONE_APP_PORT", "5151")),
        remote=True,
    )
    print(f"FiftyOne app running at http://localhost:{session.server_port}")
    session.wait()


if __name__ == "__main__":
    main()
