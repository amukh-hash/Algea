import pytest

from backend.app.data.options.chain_schema import validate_chain_rows


def test_chain_schema_validation_missing_field():
    row = {"asof": "x"}
    with pytest.raises(ValueError):
        validate_chain_rows([row])
