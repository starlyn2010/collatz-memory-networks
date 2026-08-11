#!/bin/bash
# Driver secuencial exp5: A (2 workers paralelos) -> B1 -> B2 -> C.
LOG=/home/starlyn/Escritorio/Redes\ liquidas/collatz_memory_networks/outputs/exp5_driver.log
cd /home/starlyn/Escritorio/Redes\ liquidas/collatz_memory_networks || exit 1

step() { echo "[$(date +%H:%M:%S)] $1" >> "$LOG"; }

step "INICIO exp5"
step "Lanzando A partition 0 y 1"
nohup python3 experiments/exp5_probe.py --task A --partition 0 > /dev/null 2>&1 &
P0=$!
nohup python3 experiments/exp5_probe.py --task A --partition 1 > /dev/null 2>&1 &
P1=$!
wait $P0
step "A partition 0 terminado"
wait $P1
step "A partition 1 terminado"

step "B1"
python3 experiments/exp5_probe.py --task B1 >> "$LOG" 2>&1
step "B1 terminado"

step "B2"
python3 experiments/exp5_probe.py --task B2 >> "$LOG" 2>&1
step "B2 terminado"

step "C"
python3 experiments/exp5_probe.py --task C >> "$LOG" 2>&1
step "C terminado"

step "FIN exp5"
