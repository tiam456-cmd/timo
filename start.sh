echo "🟡 Starting Xvfb on display :99..."
Xvfb :99 -screen 0 1024x768x24 &

# Wait for Xvfb to initialize
sleep 2

# Check that Xvfb is running
if pgrep Xvfb > /dev/null; then
  echo "✅ Xvfb is running on DISPLAY=:99"
else
  echo "❌ Xvfb failed to start"
  exit 1
fi

# Set the DISPLAY environment variable
export DISPLAY=:99

# Start your FastAPI app
echo "🚀 Starting FastAPI server..."
exec uvicorn gpt:app --host 0.0.0.0 --port $PORT
