"""Reject wall comparisons in which prefill safety limits throttle decode."""

def validate_decode_resource_scope(rows, environment):
    """These optional memory controls must stay above both arms' decode plans.

    Phase dispatch counts can include several plans, so this deliberately
    fails conservatively rather than accepting an ambiguous timing claim.
    """
    controls=('FUSOR_PROFILE_LARGE_PLAN_INFLIGHT_CHUNKS',
              'FUSOR_PROFILE_LARGE_PLAN_SUBMIT_SIZE')
    if not any(name in environment for name in controls):
        return
    threshold=int(environment.get('FUSOR_PROFILE_LARGE_PLAN_MIN_DISPATCHES','1024'))
    if threshold <= 0:
        raise ValueError('invalid prefill resource threshold')
    for row in rows:
        if row['phase']=='decode' and row['dispatches'] >= threshold:
            raise ValueError(f"decode has {row['dispatches']} dispatches, crossing the {threshold} prefill resource threshold; comparison is confounded")
