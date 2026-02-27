from backend.app.ml_platform.models.itransformer.model import ITransformerModel


def test_itransformer_forward_shapes():
    m = ITransformerModel()
    scores, regime = m.signal([[0.1, 0.2], [0.2, 0.3]])
    assert len(scores) == 2
    assert regime >= 0
