import ast
import unittest
from pathlib import Path

CONTROL_DIR = Path(__file__).resolve().parents[1] / "space_grasp" / "control"
FORBIDDEN = {
    "get_world_poses",
    "get_world_pose",
    "get_linear_velocity",
    "get_velocities",
    "cheat_position",
}


class NoOracleDependencyTests(unittest.TestCase):
    def test_control_package_has_no_oracle_apis(self) -> None:
        offenders: list[str] = []
        for path in CONTROL_DIR.glob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                name = getattr(node, "id", None) or getattr(node, "attr", None)
                if name in FORBIDDEN:
                    offenders.append(f"{path.name}:{getattr(node, 'lineno', 0)}:{name}")
        self.assertEqual(offenders, [], msg="control/ must not call simulator target oracles")


if __name__ == "__main__":
    unittest.main()
