from backend.app.allocator.sleeve_allocator import allocate_sleeve_gross


def test_sleeve_allocator_constraints():
    out = allocate_sleeve_gross({"vrp": {"expected_return_proxy": 10}, "selector": {"expected_return_proxy": 1}}, sleeve_max=0.6)
    assert all(v <= 0.6 for v in out.values())
