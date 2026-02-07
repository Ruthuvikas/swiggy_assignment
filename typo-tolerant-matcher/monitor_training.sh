#!/bin/bash
while true; do
    if ! ps aux | grep -q "[t]rain_transformer"; then
        echo "Training completed at $(date)"
        grep "Best validation" training_transformer.log
        break
    fi
    sleep 30
done
