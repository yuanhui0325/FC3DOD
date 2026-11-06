"""Existing G-PCC geometry backend, with checked argument-list subprocesses.

The codec options retain the original G-PCC v12 interface. This backend is
not the paper's projected/differential CABAC geometry encoder.
"""
import os
import shutil
import subprocess
from pathlib import Path


def _binary():
    value = os.environ.get('FC3DOD_TMC3')
    binary = value or str(Path(__file__).resolve().parent / 'tmc3')
    resolved = shutil.which(binary)
    if not resolved:
        raise FileNotFoundError('G-PCC tmc3 unavailable. Set FC3DOD_TMC3 to a compatible executable.')
    return resolved


def _run(options, show):
    result = subprocess.run([_binary()] + options, check=True,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if show:
        print(result.stdout, end='')
    return result


def gpcc_encode(filedir, bin_dir, show=False):
    source, output = Path(filedir).resolve(), Path(bin_dir).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if source == output:
        raise ValueError('Input and output must differ.')
    output.parent.mkdir(parents=True, exist_ok=True)
    _run(['--mode=0', '--positionQuantizationScale=1', '--trisoupNodeSizeLog2=0',
          '--neighbourAvailBoundaryLog2=8', '--intra_pred_max_node_size_log2=6',
          '--inferredDirectCodingMode=0', '--maxNumQtBtBeforeOt=4',
          '--uncompressedDataPath=' + str(source), '--compressedStreamPath=' + str(output)], show)
    if not output.is_file() or output.stat().st_size == 0:
        raise RuntimeError('tmc3 did not produce a nonempty geometry stream.')


def gpcc_decode(bin_dir, rec_dir, show=False):
    source, output = Path(bin_dir).resolve(), Path(rec_dir).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if source == output:
        raise ValueError('Input and output must differ.')
    output.parent.mkdir(parents=True, exist_ok=True)
    _run(['--mode=1', '--compressedStreamPath=' + str(source),
          '--reconstructedDataPath=' + str(output), '--outputBinaryPly=0'], show)
    if not output.is_file():
        raise RuntimeError('tmc3 did not produce a reconstructed PLY.')
