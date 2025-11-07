#!/bin/bash
# Automated script to wait for OWT tokenizer training to complete, then encode the data

cd /shared_space/jiangjiajun/workspace/cs336/assignment1-basics

echo "======================================"
echo "OWT Training & Encoding Automation"
echo "======================================"
echo ""

# Wait for training to complete
echo "⏳ Waiting for tokenizer training to complete..."
echo "   Checking for output files: data/tokenizers/owt_vocab.json"
echo ""

while true; do
    # Check if training process is still running
    if ps aux | grep -q "[p]ython.*train_owt_tokenizer"; then
        echo -ne "\r[$(date +%H:%M:%S)] Training in progress... (checking every 30s)"
        sleep 30
    else
        # Process stopped, check if files were created
        if [ -f "data/tokenizers/owt_vocab.json" ] && [ -f "data/tokenizers/owt_merges.txt" ]; then
            echo -e "\n"
            echo "✅ Training COMPLETED!"
            echo ""
            ls -lh data/tokenizers/owt* | awk '{print "  " $9 ": " $5}'
            echo ""
            break
        else
            echo -e "\n"
            echo "❌ Training process stopped but output files not found!"
            echo "   Please check for errors and restart training."
            exit 1
        fi
    fi
done

# Start encoding
echo "======================================"
echo "Starting OWT Data Encoding"
echo "======================================"
echo ""
echo "🚀 This will take approximately 1-2 hours..."
echo ""

# Run encoding in foreground so we can see the output
uv run python cs336_basics/bpe/applications/encode_owt.py

echo ""
echo "======================================"
echo "✅ Complete Pipeline Finished!"
echo "======================================"
echo ""
echo "Output files:"
ls -lh data/tokenizers/owt* data/encoded/owt* 2>/dev/null | awk '{print "  " $9 ": " $5}'

