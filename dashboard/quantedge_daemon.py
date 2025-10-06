"""
QuantEdge Daemon Runner - Keep Auto Trading Running All Day
Pure Python solution for continuous operation
"""
import subprocess
import time
import os
import signal
import sys
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - QuantEdge Daemon - %(message)s',
    handlers=[
        logging.FileHandler('quantedge_daemon.log'),
        logging.StreamHandler()
    ]
)

class QuantEdgeDaemon:
    """Daemon to keep QuantEdge running continuously"""
    
    def __init__(self):
        self.process = None
        self.running = True
        self.restart_count = 0
        
        # Configuration
        self.quantedge_dir = "/Users/andresbanuelos/PycharmProjects/QuantEdge/quantedge"
        self.dashboard_dir = os.path.join(self.quantedge_dir, "dashboard")
        self.venv_path = os.path.join(self.quantedge_dir, ".venv", "bin", "activate")
        
    def signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logging.info(f"🛑 Received signal {signum}. Shutting down QuantEdge...")
        self.running = False
        if self.process:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
        sys.exit(0)
    
    def start_quantedge(self):
        """Start the QuantEdge Streamlit app"""
        try:
            # Change to dashboard directory
            os.chdir(self.dashboard_dir)
            
            # Prepare command
            if os.path.exists(self.venv_path):
                # Use virtual environment
                cmd = [
                    "bash", "-c",
                    f"source {self.venv_path} && streamlit run quantedge_main.py --server.port 8501 --server.headless true"
                ]
                logging.info("🔄 Using virtual environment")
            else:
                # Use system Python
                cmd = [
                    "streamlit", "run", "quantedge_main.py",
                    "--server.port", "8501",
                    "--server.headless", "true"
                ]
                logging.info("⚠️  Using system Python (no venv found)")
            
            logging.info("🚀 Starting QuantEdge Auto Trading Platform...")
            
            # Start the process
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            return True
            
        except Exception as e:
            logging.error(f"❌ Failed to start QuantEdge: {e}")
            return False
    
    def monitor_process(self):
        """Monitor the QuantEdge process and restart if needed"""
        if not self.process:
            return False
        
        # Check if process is still running
        if self.process.poll() is not None:
            # Process has terminated
            return_code = self.process.returncode
            logging.warning(f"❌ QuantEdge process terminated with code {return_code}")
            return False
        
        return True
    
    def run(self):
        """Main daemon loop"""
        # Setup signal handlers
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        logging.info("🚀 QuantEdge Daemon starting...")
        logging.info(f"📍 Project directory: {self.quantedge_dir}")
        logging.info(f"🕐 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S PST')}")
        logging.info("🎯 Auto trading will run continuously during market hours")
        logging.info("📊 Access at: http://localhost:8501")
        
        while self.running:
            try:
                # Start QuantEdge if not running
                if not self.process or not self.monitor_process():
                    self.restart_count += 1
                    logging.info(f"▶️  Starting QuantEdge (attempt #{self.restart_count}) at {datetime.now().strftime('%H:%M:%S PST')}")
                    
                    if self.start_quantedge():
                        logging.info("✅ QuantEdge started successfully!")
                        
                        # Give it time to initialize
                        time.sleep(30)
                        
                        # Check if it's actually running
                        if self.monitor_process():
                            logging.info("🟢 QuantEdge is running and healthy")
                        else:
                            logging.error("❌ QuantEdge failed to start properly")
                    else:
                        logging.error("❌ Failed to start QuantEdge")
                        time.sleep(60)  # Wait before retry
                
                # Sleep and check again
                time.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logging.error(f"❌ Daemon error: {e}")
                time.sleep(60)
        
        logging.info("🛑 QuantEdge Daemon shutting down")

if __name__ == "__main__":
    print("🚀 QuantEdge Daemon - Keep Auto Trading Running All Day")
    print("📊 This will keep your trading platform running continuously")
    print("🎯 Auto trading will execute during market hours")
    print("🕐 Started at:", datetime.now().strftime('%Y-%m-%d %H:%M:%S PST'))
    print("📍 Access at: http://localhost:8501")
    print("🛑 Press Ctrl+C to stop")
    print("-" * 60)
    
    daemon = QuantEdgeDaemon()
    daemon.run()