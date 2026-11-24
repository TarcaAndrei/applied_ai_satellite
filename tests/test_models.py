import torch

from sat.models import n_parameters, resnet18


def test_resnet18_shapes_and_stem():
    model = resnet18(in_channels=12, n_classes=10).eval()
    x = torch.randn(2, 12, 64, 64)
    assert model(x).shape == (2, 10)
    # Adapted stem: no down-sampling before layer1, so the last stage sees 8x8.
    feats = torch.nn.Sequential(model.conv1, model.bn1, model.relu, model.maxpool,
                                model.layer1, model.layer2, model.layer3, model.layer4)(x)
    assert feats.shape == (2, 512, 8, 8)
    assert 11_000_000 < n_parameters(model) < 11_500_000


def test_resnet18_band_count():
    assert resnet18(in_channels=13)(torch.randn(1, 13, 64, 64)).shape == (1, 10)
