#!/bin/bash
# Quick status checker for OWT pipeline

cd /shared_space/jiangjiajun/workspace/cs336/assignment1-basics

echo "======================================"
echo "OWT Pipeline Status"
echo "======================================"
echo ""
echo "📊 Current Time: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# Check training process
if ps aux | grep -q "[p]ython.*train_owt_tokenizer"; then
    echo "1️⃣ Training Status: ✅ RUNNING"
    TRAIN_PID=$(ps aux | grep "[p]ython.*train_owt_tokenizer" | awk '{print $2}')
    TRAIN_TIME=$(ps -p $TRAIN_PID -o etime= | tr -d ' ')
    TRAIN_CPU=$(ps aux | grep "[p]ython.*train_owt_tokenizer" | awk '{print $3}')
    TRAIN_MEM=$(ps aux | grep "[p]ython.*train_owt_tokenizer" | awk '{printf "%.1f GB", $6/1024/1024}')
    echo "   PID: $TRAIN_PID"
    echo "   Runtime: $TRAIN_TIME"
    echo "   CPU: ${TRAIN_CPU}%"
    echo "   Memory: $TRAIN_MEM"
elif [ -f "data/tokenizers/owt_vocab.json" ]; then
    echo "1️⃣ Training Status: ✅ COMPLETED"
    ls -lh data/tokenizers/owt* 2>/dev/null | awk '{print "   " $9 ": " $5}'
else
    echo "1️⃣ Training Status: ⏸️  NOT RUNNING"
fi

echo ""

# Check encoding process
if ps aux | grep -q "[p]ython.*encode_owt"; then
    echo "2️⃣ Encoding Status: ✅ RUNNING"
    ENCODE_PID=$(ps aux | grep "[p]ython.*encode_owt" | awk '{print $2}')
    ENCODE_TIME=$(ps -p $ENCODE_PID -o etime= | tr -d ' ')
    ENCODE_CPU=$(ps aux | grep "[p]ython.*encode_owt" | awk '{print $3}')
    ENCODE_MEM=$(ps aux | grep "[p]ython.*encode_owt" | awk '{printf "%.1f GB", $6/1024/1024}')
    echo "   PID: $ENCODE_PID"
    echo "   Runtime: $ENCODE_TIME"
    echo "   CPU: ${ENCODE_CPU}%"
    echo "   Memory: $ENCODE_MEM"
elif [ -f "data/encoded/owt_train.npy" ]; then
    echo "2️⃣ Encoding Status: ✅ COMPLETED"
    ls -lh data/encoded/owt* 2>/dev/null | awk '{print "   " $9 ": " $5}'
else
    echo "2️⃣ Encoding Status: ⏳ WAITING (will start after training)"
fi

echo ""

# Check automation script
if ps aux | grep -q "[a]uto_train_and_encode_owt.sh"; then
    echo "🤖 Automation: ✅ ACTIVE"
else
    echo "🤖 Automation: ⏸️  INACTIVE"
fi

echo ""
echo "======================================"
echo "📝 Recent log (last 5 lines):"
echo "======================================"
tail -5 owt_pipeline.log 2>/dev/null || echo "No log file yet"
echo ""
echo "💡 Tips:"
echo "   - View full log: tail -f owt_pipeline.log"
echo "   - Expected training time: 20-40 minutes"
echo "   - Expected encoding time: 1-2 hours"
echo "======================================"

