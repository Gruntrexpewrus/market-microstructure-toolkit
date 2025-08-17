import pytest
import market_microstructure_toolkit.exchange as ex_mod


class DummyExchange:
    def __init__(self, symbols=None):
        self.symbols = symbols or []
        self.options = {}
        self.id = "dummy"

    def load_markets(self):
        return True


def test_make_exchange_sets_options(monkeypatch):
    # Patch ccxt.bybit to our dummy class
    monkeypatch.setattr(ex_mod.ccxt, "bybit", lambda kwargs: DummyExchange())
    ex = ex_mod.make_exchange("bybit", default_type="swap")
    assert isinstance(ex, DummyExchange)
    assert ex.options["defaultType"] == "swap"


def test_assert_symbol_multi_type_found(monkeypatch):
    # Force bybit to return BTC/USDT in symbols
    monkeypatch.setattr(
        ex_mod.ccxt, "bybit", lambda kwargs: DummyExchange(symbols=["BTC/USDT"])
    )
    ex, mtype = ex_mod.assert_symbol_multi_type("bybit", "BTC/USDT")
    assert isinstance(ex, DummyExchange)
    assert mtype == "spot"  # first tried type


def test_assert_symbol_multi_type_not_found(monkeypatch):
    # Force bybit to have no symbols
    monkeypatch.setattr(ex_mod.ccxt, "bybit", lambda kwargs: DummyExchange(symbols=[]))
    with pytest.raises(ValueError):
        ex_mod.assert_symbol_multi_type("bybit", "BTC/USDT")
