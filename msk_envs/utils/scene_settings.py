from dataclasses import dataclass


@dataclass
class SceneSettings:
    lanes: bool = False
    """ whether to render lane lines """

    lane_width: float = 0.0
    """ the width of lane lines, only used when lanes=True """

    meter_markers: bool = False
    """ whether to render meter markers every 10m """

    axes: bool = False
    """ whether to render axes """

    def to_dict(self):
        return {
            "lanes": self.lanes,
            "lane_width": self.lane_width,
            "meter_markers": self.meter_markers,
            "axes": self.axes
        }

    @staticmethod
    def from_dict(d: dict):
        return SceneSettings(
            lanes=d.get("lanes", False),
            lane_width=d.get("lane_width", 0.0),
            meter_markers=d.get("meter_markers", False),
            axes=d.get("axes", False)
        )
