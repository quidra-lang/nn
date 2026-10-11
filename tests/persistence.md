# Persistence fixtures

Run `python3 tests/checkpoint_format5.py /path/to/quidra` with a compatible
Core and the NN/Math package siblings. The Python oracle independently
encodes the line protocol/checksum and rounds exact decimal rationals to
IEEE value bits. Its Quidra fixture supplies nested, array, private and tuple
paths. It compares all four Parameter and State dtypes, Adam moments,
per-slot steps/corrections, canonical entry order, reordered input entries,
negative zero, subnormals, largest finite values and rounding boundaries.
It checks exact diagnostic messages for model schema differences and
unchanged checkpoint bytes and pending gradients after rejected loads.
Published formats 2 and 3 are encoded with their legacy tokens and counts;
format 4 is rejected. `tests/persistence_failures.sh` retains the existing
structural corruption cases on the new format.

The format oracle runs from the ordinary integration suite. It may also be
run with `--device N`; use `--host64` on a device without real64 arithmetic.
These commands are fixture entry points, not a claim of executed results.
Compatibility with files produced by a historical NN binary must also be
checked with that paired binary before release.

The separate `mixed_precision_master.qui DIR` fixture exercises
explicit real32 originals with real16 and real16b forward casts. Its expected
file describes the contract. It requires differentiable floating tensor
casts and must be run with the corresponding Core implementation. It is
kept separate from the ordinary training example and from persistence tests
that do not need a tracked cast.

The raw low precision hooks live in `native.qui` and use `tensor<nat32>` to
carry IEEE encodings without exposing exceptional values as Quidra real values.
`half_probe_words` operations 1/2 widen/narrow, 3/4/5/6 add/multiply/divide/sqrt.
A rejected narrowing writes 65536. `tests/half_probe.qui` compares each Metal
word with the CPU hook. Its default operation count is ten million; narrowing
can be sharded by `--begin` and `--count` over the complete binary32 domain.
`tests/half_conversions.cpp` independently compares NN's integer routines with
Core's public `quidra/abi/half_format.hpp`; it accepts an optional half-open
binary32 interval, defaulting to all 2^32 encodings for each storage format.
NaNs are compared by class because Quidra exposes no real NaN payload. Finite
values, infinities, signed zeros and narrowing rejection are exact predicates.
The fixtures are source additions awaiting the combined validation.
