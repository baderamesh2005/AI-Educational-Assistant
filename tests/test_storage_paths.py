import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_IMPORT = """
import json
from backend import config
print(json.dumps({
    "base": str(config.BASE_DIR),
    "storage": str(config.APP_STORAGE_DIR),
    "uploads": str(config.UPLOADS_DIR),
    "extracted": str(config.EXTRACTED_DIR),
    "chunks": str(config.CHUNKS_DIR),
    "vector_store": str(config.VECTOR_STORE_DIR),
    "directories_exist": all(path.is_dir() for path in (
        config.UPLOADS_DIR,
        config.EXTRACTED_DIR,
        config.CHUNKS_DIR,
        config.VECTOR_STORE_DIR,
    )),
}))
"""


def load_storage_paths(storage_root=None, create_directories=True):
    environment = os.environ.copy()
    environment.pop("APP_STORAGE_DIR", None)
    if storage_root is not None:
        environment["APP_STORAGE_DIR"] = str(storage_root)

    code = CONFIG_IMPORT
    if not create_directories:
        code = "from pathlib import Path\nPath.mkdir = lambda *args, **kwargs: None\n" + code

    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=PROJECT_ROOT,
        env=environment,
        capture_output=True,
        check=True,
        text=True,
    )
    return json.loads(result.stdout)


class ApplicationStoragePathTests(unittest.TestCase):
    def test_default_paths_preserve_repository_local_layout(self):
        paths = load_storage_paths(create_directories=False)
        root = PROJECT_ROOT

        self.assertEqual(Path(paths["storage"]), root)
        self.assertEqual(Path(paths["uploads"]), root / "data" / "uploads")
        self.assertEqual(Path(paths["extracted"]), root / "data" / "extracted")
        self.assertEqual(Path(paths["chunks"]), root / "data" / "chunks")
        self.assertEqual(Path(paths["vector_store"]), root / "vector_store")

    def test_configured_root_redirects_all_application_data_directories(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            storage_root = Path(temp_dir) / "application-storage"
            paths = load_storage_paths(storage_root)

            self.assertEqual(Path(paths["base"]), PROJECT_ROOT)
            self.assertEqual(Path(paths["storage"]), storage_root)
            self.assertEqual(Path(paths["uploads"]), storage_root / "data" / "uploads")
            self.assertEqual(Path(paths["extracted"]), storage_root / "data" / "extracted")
            self.assertEqual(Path(paths["chunks"]), storage_root / "data" / "chunks")
            self.assertEqual(Path(paths["vector_store"]), storage_root / "vector_store")
            self.assertNotEqual(storage_root, PROJECT_ROOT)

    def test_missing_configured_root_and_all_data_directories_are_created(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            storage_root = Path(temp_dir) / "not-created-yet"
            self.assertFalse(storage_root.exists())

            paths = load_storage_paths(storage_root)

            self.assertTrue(paths["directories_exist"])
            self.assertTrue(storage_root.is_dir())


if __name__ == "__main__":
    unittest.main()