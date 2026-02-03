from backend.app.risk.posture import RiskPosture

class CrashOverride:
    def __init__(self, bpi_cautious=30.0, bpi_defensive=20.0):
        self.bpi_cautious = bpi_cautious
        self.bpi_defensive = bpi_defensive

    def check(self, bpi: float, ad_slope: float) -> RiskPosture:
        """
        Returns recommended posture based on breadth.
        ad_slope: Change in AD Line over recent window (e.g. 1 hour).
        """
        if bpi < self.bpi_defensive and ad_slope < 0:
            return RiskPosture.DEFENSIVE

        if bpi < self.bpi_cautious:
            return RiskPosture.CAUTIOUS

        return RiskPosture.NORMAL
