# Correctness doctrine (user directive, binding)

We never generate invalid kernels. Every member of an e-class computes the
same value; a member that does not is a COMPILER BUG, fixed at the source
(kernel, rule, or a static legality bound in verify_plan) — never worked
around with selection changes, runtime filtering, model-side restructuring,
or cache clears.

Already enforced on branch quantized-decode-fix (merge before your final
report): production divergence in the tune race is a hard internal compiler
error; Verdict::Wrong never steers selection (fossils re-race); the
FUSOR_VERIFY_MEMBERS sweep remains the CI bug-finder that fails the build.

If you observe any wrong value from any member: stop, reduce to a repro,
fix at the source, add the case to conformance.
