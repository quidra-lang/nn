#!/usr/bin/env python3
"""Independent text-format oracle and exact/transactional persistence cases."""
import argparse
import copy
from decimal import Decimal
from fractions import Fraction
import os
from pathlib import Path
import subprocess
import tempfile

FORMATS = {"real16": (5, 10), "real16b": (8, 7), "real32": (8, 23), "real64": (11, 52)}
PATHS = {
    "real16": ["blocks[0].weight", "pair.0", "blocks[0].running"],
    "real16b": ["bfloat.weight", "pair.1", "bfloat.running"],
    "real32": ["single.weight", "single.running"],
    "real64": ["wide.weight", "wide.running"],
}


def power2(n):
    return Fraction(2 ** n) if n >= 0 else Fraction(1, 2 ** -n)


def round_even(value):
    whole, rest = divmod(value.numerator, value.denominator)
    twice = 2 * rest
    return whole + (twice > value.denominator or (twice == value.denominator and whole % 2))


def bits(text, dtype):
    """Round the exact decimal rational once to the declared IEEE format."""
    decimal = Decimal(text)
    if not decimal.is_finite():
        raise ValueError("non-finite value")
    exponent_bits, fraction_bits = FORMATS[dtype]
    sign = int(decimal.is_signed())
    value = abs(Fraction(decimal))
    width = 1 + exponent_bits + fraction_bits
    if not value:
        return sign << (width - 1)
    bias = (1 << (exponent_bits - 1)) - 1
    exponent = value.numerator.bit_length() - value.denominator.bit_length()
    if value < power2(exponent):
        exponent -= 1
    quantum = max(exponent, 1 - bias) - fraction_bits
    significand = round_even(value / power2(quantum))
    if significand >= (1 << (fraction_bits + 1)):
        exponent += 1
        significand >>= 1
    if exponent > bias:
        raise ValueError("out of range")
    encoded_exp = 0 if significand < (1 << fraction_bits) else max(exponent, 1 - bias) + bias
    fraction = significand if not encoded_exp else significand - (1 << fraction_bits)
    return (sign << (width - 1)) | (encoded_exp << fraction_bits) | fraction


def boundary_values(dtype):
    exponent_bits, fraction_bits = FORMATS[dtype]
    bias = (1 << (exponent_bits - 1)) - 1
    # Exact decimal spellings are intentionally independent of nn's writer.
    values = [Fraction(0), Fraction(0), power2(1 - bias - fraction_bits),
              power2(1 - bias), (2 - power2(-fraction_bits)) * power2(bias),
              1 + power2(-fraction_bits), 1 + power2(-fraction_bits - 1),
              1 + 3 * power2(-fraction_bits - 1)]
    result = []
    for value in values:
        # Denominators are powers of two, so this is exact finite decimal text.
        denominator = value.denominator
        places = denominator.bit_length() - 1
        numerator = value.numerator * 5 ** places
        digits = str(numerator).rjust(places + 1, "0")
        text = digits if not places else digits[:-places] + "." + digits[-places:]
        result.append(text if "." in text else text + ".0")
    result[1] = "-0.0"
    return result


def seal(lines):
    a, b = 1, 0
    for byte in ("\n".join(lines) + "\n").encode():
        a = (a + byte) % 65521
        b = (b + a) % 65521
    return lines + ["ADLER65521-V1", f"{b}:{a}"]


def entry_lines(entry, legacy=False):
    kind, dtype, path, shape, values = (entry[k] for k in ("kind", "dtype", "path", "shape", "values"))
    token = {"real32": "float32", "real64": "float"}.get(dtype, dtype)
    prefix = [kind, kind + "<" + token + ">", token, path] if legacy else [kind, dtype, path]
    return prefix + [str(len(shape))] + list(map(str, shape)) + [str(len(values))] + values


def encode(document, version="5"):
    entries = document["entries"]
    lines = ["QUIDRA_DNN_STATE", version, document["container"], document["root"]]
    if version == "5":
        lines += [str(len(entries))]
    else:
        lines += [str(sum(e["kind"] == k and e["dtype"] == t for e in entries))
                  for k, t in [("Parameter", "real32"), ("Parameter", "real64"), ("State", "real32"), ("State", "real64")]]
    for entry in entries:
        lines += entry_lines(entry, version != "5")
    if document["container"] == "model+Adam":
        lines += ["Adam"] + document["hyper"]
        slots = document["slots"]
        if version == "5":
            lines += [str(len(slots))]
        else:
            lines += slots[0]["step"][:1 if version == "2" else 3]
            lines += [str(sum(s["first"]["dtype"] == t for s in slots)) for t in ["real32", "real64"]]
        for slot in slots:
            if version == "5":
                lines += ["AdamStep", slot["path"]] + slot["step"]
            lines += entry_lines(slot["first"], version != "5")
            lines += entry_lines(slot["second"], version != "5")
    return "\n".join(seal(lines)) + "\n"


def decode(path):
    lines = path.read_text().splitlines()
    assert seal(lines[:-2]) == lines, "checksum"
    cursor = 0

    def take():
        nonlocal cursor
        value = lines[cursor]
        cursor += 1
        return value

    def entry():
        kind, dtype, path = take(), take(), take()
        shape = [int(take()) for _ in range(int(take()))]
        count = int(take())
        product = 1
        for extent in shape:
            product *= extent
        assert product == count
        values = [take() for _ in range(count)]
        return dict(kind=kind, dtype=dtype, path=path, shape=shape, values=values)

    assert take() == "QUIDRA_DNN_STATE"
    assert take() == "5"
    document = dict(container=take(), root=take())
    document["entries"] = [entry() for _ in range(int(take()))]
    if document["container"] == "model+Adam":
        assert take() == "Adam"
        document["hyper"] = [take() for _ in range(4)]
        document["slots"] = []
        for _ in range(int(take())):
            assert take() == "AdamStep"
            slot = dict(path=take(), step=[take() for _ in range(3)], first=entry(), second=entry())
            assert slot["first"]["kind"] == "AdamFirst" and slot["second"]["kind"] == "AdamSecond"
            assert slot["path"] == slot["first"]["path"] == slot["second"]["path"]
            document["slots"].append(slot)
    assert cursor == len(lines) - 2
    return document


def semantic(document):
    def entry(e):
        return e["kind"], e["dtype"], e["path"], tuple(e["shape"]), tuple(bits(v, e["dtype"]) for v in e["values"])
    result = {"root": document["root"], "container": document["container"],
              "entries": {e["path"]: entry(e) for e in document["entries"]}}
    assert len(result["entries"]) == len(document["entries"])
    if "slots" in document:
        result["hyper"] = tuple(bits(v, "real64") for v in document["hyper"])
        result["slots"] = {s["path"]: (int(s["step"][0]), tuple(bits(v, "real64") for v in s["step"][1:]), entry(s["first"]), entry(s["second"])) for s in document["slots"]}
        assert len(result["slots"]) == len(document["slots"])
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("quidra")
    parser.add_argument("--device", default="-1")
    parser.add_argument("--host64", action="store_true")
    args = parser.parse_args()
    repository = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory() as directory:
        scratch = Path(directory)
        environment = dict(os.environ, QUIDRA_PACKAGE_PATH=str(repository.parent), QUIDRA_CACHE_DIR=str(scratch / "cache"))

        def run(action, output, checkpoint, source=None, legacy=False):
            argv = [args.quidra, str(repository / ("tests/checkpoint_legacy.qui" if legacy else "tests/checkpoint_format5.qui")), action, str(output)]
            if not legacy:
                argv += ["--device", args.device]
            if checkpoint:
                argv.append("--checkpoint")
            if source:
                argv += ["--source", str(source)]
            if not legacy and args.host64:
                argv.append("--host64")
            result = subprocess.run(argv, env=environment, text=True, capture_output=True)
            assert result.returncode == 0, (argv, result.returncode, result.stdout, result.stderr)
            assert result.stderr == "", result.stderr
            return result.stdout

        for checkpoint in [False, True]:
            seed = scratch / ("checkpoint" if checkpoint else "model")
            assert run("seed", seed, checkpoint) == ""
            document = decode(seed)
            expected_order = [(kind, dtype, path) for kind in ["Parameter", "State"] for dtype in FORMATS for path in PATHS[dtype] if (path.endswith("weight")) == (kind == "Parameter")]
            assert [(e["kind"], e["dtype"], e["path"]) for e in document["entries"]] == expected_order
            for e in document["entries"]:
                assert e["shape"] == [8]
                e["values"] = boundary_values(e["dtype"])
            if checkpoint:
                assert [s["path"] for s in document["slots"]] == [e["path"] for e in document["entries"] if e["kind"] == "Parameter"]
                for slot in document["slots"]:
                    dtype = next(e["dtype"] for e in document["entries"] if e["path"] == slot["path"])
                    state_type = "real64" if dtype == "real64" else "real32"
                    assert slot["first"]["dtype"] == slot["second"]["dtype"] == state_type
                    slot["step"] = ["1", "0.1", "0.001"]
                    for component in ["first", "second"]:
                        slot[component]["values"] = boundary_values(state_type)
            source = scratch / "crafted"
            source.write_text(encode(document))
            output = scratch / "resaved"
            assert run("roundtrip", output, checkpoint, source) == ""
            assert semantic(decode(output)) == semantic(document), "value-bit round trip"
            reordered = copy.deepcopy(document)
            reordered["entries"].reverse()
            if checkpoint:
                reordered["slots"].reverse()
            source.write_text(encode(reordered))
            assert run("roundtrip", output, checkpoint, source) == ""
            assert output.read_bytes() == encode(decode(output)).encode()
            assert semantic(decode(output)) == semantic(document), "keyed order independence"

            def reject(name, bad, code, detail=None, raw=None, message=None):
                source.write_text(raw if raw is not None else encode(bad))
                stdout = run("reject", output, checkpoint, source)
                assert stdout.startswith(code + "\n"), (name, stdout)
                assert stdout.endswith("gradients unchanged true\n"), (name, stdout)
                if detail:
                    assert detail in stdout, (name, detail, stdout)
                if message is not None:
                    assert stdout == code + "\n" + message + "\ngradients unchanged true\n", (name, stdout, message)
                assert output.read_bytes() == Path(str(output) + ".before").read_bytes(), name

            for version in ["4", "999"]:
                reject("unsupported version", document, "NN_CHECKPOINT_FILE", "unsupported format version", encode(document).replace("\n5\n", "\n" + version + "\n", 1))
            for change in ["rename", "dtype", "shape", "missing", "extra", "duplicate", "root", "unknown-existing", "unknown-extra"]:
                bad = copy.deepcopy(document)
                e = bad["entries"][0]
                if change == "rename": e["path"] = "renamed[0].weight"
                elif change == "dtype": e["dtype"] = "real16b"
                elif change == "shape": e["shape"] = [2, 4]
                elif change == "missing": bad["entries"].pop(0)
                elif change == "extra": bad["entries"].append(dict(e, path="extra.weight"))
                elif change == "duplicate": bad["entries"].append(copy.deepcopy(e))
                elif change == "root": bad["root"] = "OtherModel"
                elif change == "unknown-existing": e["dtype"] = "real8"; e["values"] = ["not a numeric value"] * 8
                elif change == "unknown-extra": bad["entries"].append(dict(e, path="extra.weight", dtype="real8", values=["not a numeric value"] * 8))
                # The known different dtype uses in-range text so this reaches schema comparison.
                if change == "dtype": e["values"] = ["0.0"] * 8
                operation = "nn.Adam.load" if checkpoint else "nn.load"
                message = operation + ': "' + str(source) + '" does not match the model; nothing was loaded'
                original = document["entries"][0]
                name = original["path"]
                label = name + "  Parameter<real16> [8]"
                if change in ["rename", "missing"]:
                    message += "\n  missing from the file (1):\n    " + label
                if change == "rename":
                    message += "\n  not in the model (1):\n    renamed[0].weight  Parameter<real16> [8]\n  likely a renamed field: renamed -> blocks"
                elif change == "extra":
                    message += "\n  not in the model (1):\n    extra.weight  Parameter<real16> [8]"
                elif change == "duplicate":
                    message += "\n  not in the model (1):\n    " + label + " (duplicate path)"
                elif change in ["dtype", "unknown-existing"]:
                    message += "\n  dtype differs (1):\n    " + name + "  file Parameter<" + e["dtype"] + ">, model Parameter<real16>"
                elif change == "shape":
                    message += "\n  shape differs (1):\n    " + name + "  file [2, 4], model [8]"
                elif change == "root":
                    message += "\n  root type differs: file OtherModel, model " + document["root"]
                elif change == "unknown-extra":
                    message += "\n  not in the model (1):\n    extra.weight  Parameter<real8> [8]"
                reject(change, bad, "NN_CHECKPOINT", message=message)
            for dtype in FORMATS:
                for text in ["bad-value", "1e9999"]:
                    bad = copy.deepcopy(document)
                    entry = next(e for e in bad["entries"] if e["dtype"] == dtype)
                    entry["values"][3] = text
                    reject("invalid value", bad, "NN_CHECKPOINT_FILE", entry["path"] + "[3]")
            raw = encode(document)
            reject("checksum", document, "NN_CHECKPOINT_FILE", "checksum", "\n".join(raw.splitlines()[:-1] + ["0:0"]) + "\n")
            reject("trailing", document, "NN_CHECKPOINT_FILE", "trailing data", raw + "trailing\n")
            reject("truncated", document, "NN_CHECKPOINT_FILE", raw="\n".join(raw.splitlines()[:-4]) + "\n")
            if checkpoint:
                for component in ["first", "second"]:
                    for change in ["dtype", "shape", "path"]:
                        bad = copy.deepcopy(document)
                        moment = bad["slots"][0][component]
                        if change == "dtype": moment["dtype"] = "real16"; moment["values"] = ["0.0"] * 8
                        elif change == "shape": moment["shape"] = [2, 4]
                        else: moment["path"] = "wrong.weight"
                        reject("Adam moment " + change, bad, "NN_CHECKPOINT_FILE" if change == "path" else "NN_CHECKPOINT")
                for step in [["-1", "0.1", "0.001"], ["0", "0.1", "0.001"], ["1", "1.5", "0.001"]]:
                    bad = copy.deepcopy(document); bad["slots"][0]["step"] = step
                    reject("Adam step", bad, "NN_CHECKPOINT_FILE")
                for index, value in enumerate(["-1.0", "1.0", "1.0", "0.0"]):
                    bad = copy.deepcopy(document); bad["hyper"][index] = value
                    reject("Adam hyperparameter", bad, "NN_CHECKPOINT_FILE")
                bad = copy.deepcopy(document); bad["slots"].append(copy.deepcopy(bad["slots"][0]))
                reject("duplicate Adam slot", bad, "NN_CHECKPOINT")
                # Boundary moments prove lossless storage, not that the next
                # update fits the Parameter's dtype. Keep those roundtrip and
                # malformed-load cases above; continue a separate bounded state.
                continuation = copy.deepcopy(document)
                for slot in continuation["slots"]:
                    slot["first"]["values"] = ["0.125"] * len(slot["first"]["values"])
                    slot["second"]["values"] = ["0.25"] * len(slot["second"]["values"])
                continuation_reordered = copy.deepcopy(continuation)
                continuation_reordered["entries"].reverse()
                continuation_reordered["slots"].reverse()
                # Resume one serialized state and one reordered equivalent state;
                # the next update must produce exactly the same checkpoint bytes.
                source.write_text(encode(continuation))
                assert run("continue", scratch / "continued", True, source) == ""
                source.write_text(encode(continuation_reordered))
                assert run("continue", scratch / "continued-reordered", True, source) == ""
                assert (scratch / "continued").read_bytes() == (scratch / "continued-reordered").read_bytes()

        for checkpoint in [False, True]:
            seed = scratch / "legacy-seed"
            assert run("seed", seed, checkpoint, legacy=True) == ""
            document = decode(seed)
            assert [(e["kind"], e["dtype"], e["path"]) for e in document["entries"]] == [("Parameter", "real32", "single"), ("Parameter", "real64", "wide"), ("State", "real32", "running"), ("State", "real64", "total")]
            for entry in document["entries"]:
                entry["values"] = boundary_values(entry["dtype"])
            if checkpoint:
                for slot in document["slots"]:
                    slot["step"] = ["1", repr(1.0 - float(document["hyper"][1])), repr(1.0 - float(document["hyper"][2]))]
                    for component in ["first", "second"]:
                        slot[component]["values"] = boundary_values(slot[component]["dtype"])
            for version in (["2", "3"] if checkpoint else ["2"]):
                source = scratch / ("legacy-" + version)
                source.write_text(encode(document, version))
                output = scratch / "legacy-resaved"
                assert run("roundtrip", output, checkpoint, source, legacy=True) == ""
                assert semantic(decode(output)) == semantic(document), "legacy exact values and corrections"
                # A public-only legacy file never fills absent private/half paths.
                rejected = scratch / "legacy-rejected"
                stdout = run("reject", rejected, checkpoint, source)
                assert stdout.startswith("NN_CHECKPOINT\n") and "blocks[0].weight" in stdout and "bfloat.weight" in stdout
                assert stdout.endswith("gradients unchanged true\n")
                assert rejected.read_bytes() == Path(str(rejected) + ".before").read_bytes()
        print("nn format5 persistence: ok")


if __name__ == "__main__":
    main()
