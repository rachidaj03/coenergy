import torch

from coenergy.data import TargetScaler
from coenergy.models import ThermalNet, predict_encoded


def _scaler() -> TargetScaler:
    y = torch.rand(100, 6) * 100 + 1
    return TargetScaler().fit(y)


def test_hlc_is_sum_of_hb():
    net, sc = ThermalNet(hidden=16), _scaler()
    out = sc.decode_free(net(torch.randn(4, 2, 720)))
    assert out.shape == (4, 6)
    assert torch.allclose(out[:, 0], out[:, 1] + out[:, 2], atol=1e-4)
    assert (out > 0).all()


def test_encode_decode_roundtrip():
    sc = _scaler()
    y = torch.rand(8, 6) * 50 + 1
    z = sc.encode(y)
    back = torch.exp(z * sc.sigma + sc.mu)
    assert torch.allclose(back, y, rtol=1e-4)


def test_predict_encoded_shape():
    net, sc = ThermalNet(hidden=16), _scaler()
    assert predict_encoded(net(torch.randn(3, 2, 720)), sc).shape == (3, 6)


def test_lstm_variant_keeps_constraint():
    net, sc = ThermalNet(encoder="lstm", hidden=8), _scaler()
    out = sc.decode_free(net(torch.randn(4, 2, 120)))
    assert torch.allclose(out[:, 0], out[:, 1] + out[:, 2], atol=1e-4)


def test_aux_branches_keep_constraint():
    net, sc = ThermalNet(hidden=16, use_aux=True), _scaler()
    net.eval()
    out = sc.decode_free(net(torch.randn(4, 2, 480)))
    assert out.shape == (4, 6) and torch.isfinite(out).all()
    assert torch.allclose(out[:, 0], out[:, 1] + out[:, 2], atol=1e-4)


def test_ratio_mode_guarantees_tau_order():
    y = torch.rand(64, 6) * 50 + 5
    y[:, 5] = y[:, 3] * (torch.rand(64) * 0.8 + 0.1)
    y[:, 4] = y[:, 5] * (torch.rand(64) * 0.8 + 0.1)
    sc = TargetScaler("ratio").fit(y)
    out = sc.decode_free(torch.randn(32, 5) * 3)
    assert (out[:, 3] > out[:, 5]).all() and (out[:, 5] > out[:, 4]).all()
    assert torch.allclose(out[:, 0], out[:, 1] + out[:, 2], atol=1e-4)
    assert torch.allclose(sc.decode_free(sc.encode(y)[:, 1:])[:, 1:], y[:, 1:], rtol=1e-3, atol=1e-3)
