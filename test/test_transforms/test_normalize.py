import numpy as np
import pytest

from fluxvla.transforms.normalize import (DenormalizePrivateAction,
                                          NormalizeStatesAndActions)


def _stats(dim):
    return {
        'mean': np.zeros(dim),
        'std': np.ones(dim),
        'min': np.zeros(dim),
        'max': np.full(dim, 2.0),
        'q01': np.zeros(dim),
        'q99': np.full(dim, 2.0),
    }


def test_normalize_states_uses_active_dimension_for_longer_stats():
    transform = NormalizeStatesAndActions(
        state_key='proprio',
        action_key='action',
        state_dim=32,
        norm_type='min_max')
    data = {
        'states': np.ones(16),
        'stats': {
            'proprio': _stats(18),
            'action': _stats(18),
        },
    }

    result = transform(data)

    np.testing.assert_allclose(result['states'], np.zeros(32), atol=1e-6)


def test_denormalize_actions_uses_configured_action_dimension():
    transform = DenormalizePrivateAction(
        norm_stats={'private': {'action': _stats(18)}},
        action_dim=16,
        norm_type='min_max')
    normalized = np.zeros((50, 32))

    result = transform({'action': normalized})

    assert result.shape == (50, 16)
    np.testing.assert_allclose(result, np.ones((50, 16)))


def test_normalize_rejects_short_statistics_vector():
    transform = NormalizeStatesAndActions(
        state_key='proprio', action_key='action', norm_type='min_max')
    data = {
        'states': np.zeros(16),
        'stats': {
            'proprio': _stats(15),
            'action': _stats(15),
        },
    }

    with pytest.raises(ValueError, match='at least 16 values are required'):
        transform(data)
