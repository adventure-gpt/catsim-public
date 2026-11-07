@echo off
rem catSim as a worker for the full-state suite's own RocketSim v3 harness (score_v3.py): docs\BENCHMARKS.md.
rem Needs CATSIM_DATA (catSim's dumper output).
set PYTHONPATH=%~dp0..;%PYTHONPATH%
python -m simbench.fs_worker catsim
