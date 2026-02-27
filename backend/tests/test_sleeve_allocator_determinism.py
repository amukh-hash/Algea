from backend.app.allocator.sleeve_allocator import allocate_sleeve_gross


def test_sleeve_allocator_determinism():
    metrics = {"vrp": {"expected_return_proxy": 0.3}, "selector": {"expected_return_proxy": 0.2}}
    a = allocate_sleeve_gross(metrics)
    b = allocate_sleeve_gross(metrics)
    assert a == b
