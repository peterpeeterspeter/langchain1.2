#!/bin/bash
# Monitor CyBet production pipeline progress

LOG_FILE="cybet_production_fixed.log"

echo "🔍 Monitoring CyBet Production Pipeline..."
echo "=========================================="
echo ""

# Check if process is running
if pgrep -f "run_production_complete.py" > /dev/null; then
    echo "✅ Pipeline is RUNNING"
    echo ""
else
    echo "❌ Pipeline is NOT running"
    echo ""
fi

# Show recent progress
echo "📊 Recent Activity (last 30 lines):"
echo "-----------------------------------"
tail -30 "$LOG_FILE" 2>/dev/null | grep -v "HTTP Request\|WARNING\|UserWarning\|PyTorch\|notice\|schema_extra\|DeprecationWarning\|USER_AGENT" | tail -20

echo ""
echo "🔍 Key Metrics:"
echo "-----------------------------------"

# Check research progress
if grep -q "fields extracted" "$LOG_FILE" 2>/dev/null; then
    echo "✅ Research:"
    grep "fields extracted\|quality:" "$LOG_FILE" 2>/dev/null | tail -3
else
    echo "⏳ Research: In progress..."
fi

# Check writing progress
if grep -q "Content.*chars\|Generated.*characters" "$LOG_FILE" 2>/dev/null; then
    echo ""
    echo "✅ Writing:"
    grep "Content.*chars\|Generated.*characters" "$LOG_FILE" 2>/dev/null | tail -2
else
    echo ""
    echo "⏳ Writing: Waiting for research..."
fi

# Check image progress
if grep -q "images.*uploaded\|Image Agent.*Processed" "$LOG_FILE" 2>/dev/null; then
    echo ""
    echo "✅ Images:"
    grep "images.*uploaded\|Image Agent.*Processed" "$LOG_FILE" 2>/dev/null | tail -2
else
    echo ""
    echo "⏳ Images: Waiting..."
fi

# Check affiliate progress
if grep -q "affiliate links\|Inserted.*links" "$LOG_FILE" 2>/dev/null; then
    echo ""
    echo "✅ Affiliate:"
    grep "affiliate links\|Inserted.*links" "$LOG_FILE" 2>/dev/null | tail -2
else
    echo ""
    echo "⏳ Affiliate: Waiting..."
fi

# Check publishing progress
if grep -q "Post ID\|published successfully\|Post published" "$LOG_FILE" 2>/dev/null; then
    echo ""
    echo "✅ Publishing:"
    grep "Post ID\|published successfully\|Post published" "$LOG_FILE" 2>/dev/null | tail -2
else
    echo ""
    echo "⏳ Publishing: Waiting..."
fi

# Check for errors
if grep -q "ERROR\|Exception\|Traceback\|Failed\|failed" "$LOG_FILE" 2>/dev/null; then
    echo ""
    echo "⚠️  Errors Found:"
    grep -E "ERROR|Exception|Traceback|Failed|failed" "$LOG_FILE" 2>/dev/null | grep -v "HTTP Request" | tail -5
fi

echo ""
echo "📝 Full log: tail -f $LOG_FILE"
echo ""


