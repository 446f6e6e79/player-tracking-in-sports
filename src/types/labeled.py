_TEAM_COLORS_BGR = {
    "ball":     (0, 255, 255),    # yellow
    "red":      (0, 0, 255),      # red
    "white":    (255, 255, 255),  # white
    "referee":  (0, 165, 255),    # orange
}
_FALLBACK_COLOR = (128, 128, 128)


def _split_label(class_name: str) -> tuple[str, str]:
    """Split 'Red_11' -> ('red', '11'); 'Ball' -> ('ball', '')."""
    parts = class_name.split("_", 1)
    head = parts[0].lower()
    number = parts[1] if len(parts) == 2 else ""
    # The fine-tuned model spells it 'Refree'; accept 'Referee' too.
    if head == "referee":
        head = "refree"
    return head, number


class LabeledObject:
    """
    Base class for labeled objects in our dataset, providing common parsing and properties.
    Subclasses must provide a class_name field (e.g. "White_14", "Red_11", "Ball")
    """

    class_name: str  # provided by subclass; declared for static analysis only

    @property
    def team(self) -> str:
        """One of 'red', 'white', 'referee', 'ball', or 'unknown'."""
        head, _ = _split_label(self.class_name)
        if head in ("red", "white", "ball"):
            return head
        if head == "refree":
            return "referee"
        return "unknown"

    @property
    def number(self) -> str:
        """Jersey number string. Empty for ball; raw class_name for unknown labels."""
        head, number = _split_label(self.class_name)
        if head == "ball":
            return ""
        if head in ("red", "white", "refree"):
            return number
        return self.class_name

    @property
    def color(self) -> tuple[int, int, int]:
        """BGR color for this label. Gray fallback for unknown labels."""
        team = self.team
        if team == "unknown":
            return _FALLBACK_COLOR
        return _TEAM_COLORS_BGR[team]

    @property
    def is_ball(self) -> bool:
        return self.team == "ball"
