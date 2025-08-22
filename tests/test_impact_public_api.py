# Created by LP
# Date: 2025-08-22
# Trade only the money you can't afford to lose
# Then go back to the mine
# And try again.
# This was coded with love <3

# Very light "public API" smoke checks:
# - the modules import
# - the dataclasses expose the minimal expected fields
# - no logging/filesystem assumptions

from __future__ import annotations

import importlib


def test_import_modules() -> None:
    for mod in [
        "market_microstructure_toolkit.impact",
        "market_microstructure_toolkit.impact_twap",
        "market_microstructure_toolkit.impact_vwap",
    ]:
        importlib.import_module(mod)


def test_configs_have_core_fields() -> None:
    it = importlib.import_module("market_microstructure_toolkit.impact_twap")
    iv = importlib.import_module("market_microstructure_toolkit.impact_vwap")

    for C in [getattr(it, "TWAPConfig"), getattr(iv, "VwapConfig")]:
        fields = getattr(C, "__dataclass_fields__", {})
        for must in ["side", "target_qty", "slices", "fee_bps"]:
            assert must in fields
        # allow either depth or depth_cap
        assert ("depth" in fields) or ("depth_cap" in fields)
