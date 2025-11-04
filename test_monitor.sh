#!/bin/bash
# Monitor CyBet production test with fixes

LOG_FILE="cybet_production_test.log"

echo "🔍 Monitoring CyBet Production Test (with Document Storage Fix)"
echo "================================================================"
echo ""

# Check if process is running
if pgrep -f "run_production_complete.py" > /dev/null; then
    echo "✅ Pipeline is RUNNING"
    echo ""
else
    echo "❌ Pipeline is NOT running (may have completed)"
    echo ""
fi

# Show recent progress
echo "📊 Recent Activity:"
echo "-------------------"
tail -50 "$LOG_FILE" 2>/dev/null | strings | grep -E "(Research|documents|fields|stored|Supabase|RAG|Writing|Content|chunks|CyBet|ERROR)" | tail -20

echo ""
echo "🔍 Key Metrics:"
echo "-------------------"

# Check document collection
if grep -q "documents collected" "$LOG_FILE" 2>/dev/null; then
    echo "✅ Document Collection:"
    grep "documents collected\|Found.*documents" "$LOG_FILE" 2>/dev/null | tail -3
else
    echo "⏳ Document Collection: In progress..."
fi

# Check Supabase storage
if grep -q "Stored.*chunks\|stored.*Supabase" "$LOG_FILE" 2>/dev/null; then
    echo ""
    echo "✅ Supabase Storage:"
    grep "Stored.*chunks\|stored.*Supabase" "$LOG_FILE" 2>/dev/null | tail -2
else
    echo ""
    echo "⏳ Supabase Storage: Waiting for documents..."
fi

# Check fields extracted
if grep -q "fields extracted\|fields_populated" "$LOG_FILE" 2>/dev/null; then
    echo ""
    echo "✅ Fields Extracted:"
    grep "fields extracted\|fields_populated" "$LOG_FILE" 2>/dev/null | tail -2
else
    echo ""
    echo "⏳ Fields Extraction: In progress..."
fi

# Check RAG retrieval
if grep -q "retriev\|RAG\|similarity" "$LOG_FILE" 2>/dev/null; then
    echo ""
    echo "✅ RAG Retrieval:"
    grep -i "retriev\|RAG\|similarity" "$LOG_FILE" 2>/dev/null | tail -2
else
    echo ""
    echo "⏳ RAG Retrieval: Waiting for research..."
fi

# Check content generation
if grep -q "Content.*chars\|Generated.*characters" "$LOG_FILE" 2>/dev/null; then
    echo ""
    echo "✅ Content Generation:"
    grep "Content.*chars\|Generated.*characters" "$LOG_FILE" 2>/dev/null | tail -2
else
    echo ""
    echo "⏳ Content Generation: Waiting..."
fi

# Check for CyBet-specific content
if grep -q "CyBet\|cybet" "$LOG_FILE" 2>/dev/null; then
    echo ""
    echo "✅ CyBet Mentions Found:"
    grep -i "cybet" "$LOG_FILE" 2>/dev/null | grep -v "Research Agent\|Starting\|Query:" | head -3
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


