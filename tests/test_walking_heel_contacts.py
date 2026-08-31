import unittest
from pathlib import Path
from xml.etree import ElementTree

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = REPO_ROOT / "msk_envs/msk_models/sprinter/sprinter_model.osim"
CONTACT_PATH = REPO_ROOT / "msk_envs/msk_models/sprinter/contact_params/contact_params.yaml"


class WalkingHeelContactTest(unittest.TestCase):
    def test_both_heel_spheres_use_aligned_walking_geometry(self):
        root = ElementTree.parse(MODEL_PATH).getroot()
        contacts = {contact.attrib["name"]: contact for contact in root.iter("ContactSphere")}

        for side in ("right", "left"):
            heel = contacts[f"{side}_foot_7"]
            self.assertEqual(heel.findtext("location"), "0.01 0.0018 0")
            self.assertEqual(float(heel.findtext("radius")), 0.05)

    def test_both_heels_use_walking_contact_material(self):
        with CONTACT_PATH.open() as stream:
            contacts = yaml.safe_load(stream)

        for side in ("right", "left"):
            heel = contacts[f"{side}_foot_7"]
            self.assertEqual(heel["stiffness"], 500000.0)
            self.assertEqual(heel["static_friction"], 0.8)
            self.assertEqual(heel["dynamic_friction"], 0.8)


if __name__ == "__main__":
    unittest.main()
