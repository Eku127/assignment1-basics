#!/bin/bash
# Script to monitor OWT tokenizer training progress

echo "======================================"
echo "OWT Tokenizer Training Monitor"
echo "======================================"
echo ""

# Check if process is running
if ps aux | grep "train_owt_tokenizer" | grep -v grep > /dev/null; then
    echo "✅ Training process is RUNNING"
    echo ""
    
    # Show process details
    echo "Process details:"
    ps aux | grep "train_owt_tokenizer" | grep -v grep | awk '{printf "  PID: %s\n  CPU: %s%%\n  MEM: %s%%\n  Runtime: %s\n", $2, $3, $4, $10}'
    echo ""
    
    # Check memory usage
    echo "Memory usage:"
    ps aux | grep "[p]ython.*train_owt_tokenizer" | awk '{print "  " $6/1024 " MB"}'
    echo ""
    
    # Check if output files exist
    if [ -d "data/tokenizers" ]; then
        echo "Output directory exists:"
        ls -lh data/tokenizers/ | tail -n +2 | awk '{print "  " $9 " (" $5 ")"}'
    else
        echo "⏳ Still processing... Output directory not created yet"
    fi
else
    echo "❌ Training process NOT running"
    echo ""
    
    # Check if training completed
    if [ -f "data/tokenizers/owt_vocab.json" ]; then
        echo "✅ Training COMPLETED! Files created:"
        ls -lh data/tokenizers/owt* 2>/dev/null | awk '{print "  " $9 " (" $5 ")"}'
    else
        echo "Training may have failed or not started yet"
    fi
fi

echo ""
echo "======================================"

