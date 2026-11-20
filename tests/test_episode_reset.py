import unittest

from space_grasp.config import load_experiment_config


class EpisodeResetConfigTests(unittest.TestCase):
    def test_settling_frames_are_positive(self) -> None:
        config = load_experiment_config()
        self.assertGreater(int(config.episode.settling_frames), 0)
        self.assertGreater(float(config.episode.timeout_s), 0.0)
        self.assertGreaterEqual(int(config.episode.max_retries), 1)


if __name__ == "__main__":
    unittest.main()
