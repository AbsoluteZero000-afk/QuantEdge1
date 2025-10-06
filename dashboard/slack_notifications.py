"""
QuantEdge Slack Notification System - FIXED VERSION
Looks for .env file in multiple locations with robust environment loading
"""
import json
import requests
import os
from datetime import datetime
import logging
from typing import Dict, Optional, List
from pathlib import Path

logger = logging.getLogger(__name__)

class SlackNotificationSystem:
    """Advanced Slack notification system for QuantEdge trading platform"""
    
    def __init__(self):
        """Initialize Slack notification system with robust .env loading"""
        self.webhook_url = None
        self.enabled = False
        
        # Try to find and load .env file from multiple locations
        self._load_environment_variables()
        
        # Debug logging
        print(f"🔍 Slack Debug Info:")
        print(f"   Webhook URL found: {'Yes' if self.webhook_url else 'No'}")
        if self.webhook_url:
            print(f"   Webhook URL length: {len(self.webhook_url)} chars")
            print(f"   Webhook starts with: {self.webhook_url[:50] if len(self.webhook_url) > 50 else self.webhook_url}...")
        print(f"   Slack enabled: {self.enabled}")
        
        if not self.enabled:
            logger.warning("⚠️ Slack webhook URL not found")
            print("❌ Slack webhook URL not found")
        else:
            logger.info("✅ Slack notifications enabled")
            print("✅ Slack notifications enabled")
    
    def _load_environment_variables(self):
        """Load environment variables from .env file in multiple locations"""
        print("🔍 Searching for .env file...")
        
        # Possible .env file locations
        possible_locations = [
            # Current working directory
            ".env",
            # Dashboard directory
            "dashboard/.env",
            # Parent directory
            "../.env", 
            # Absolute paths
            "/Users/andresbanuelos/PycharmProjects/QuantEdge/.env",
            "/Users/andresbanuelos/PycharmProjects/QuantEdge/dashboard/.env",
            # Current script directory
            Path(__file__).parent / ".env",
            Path(__file__).parent.parent / ".env"
        ]
        
        env_found = False
        
        for location in possible_locations:
            env_path = Path(location)
            print(f"   Checking: {env_path.absolute()}")
            
            if env_path.exists() and env_path.is_file():
                print(f"✅ Found .env file at: {env_path.absolute()}")
                env_found = True
                
                # Read the .env file manually
                try:
                    with open(env_path, 'r') as f:
                        for line in f:
                            line = line.strip()
                            if line and not line.startswith('#') and '=' in line:
                                key, value = line.split('=', 1)
                                key = key.strip()
                                value = value.strip().strip('"').strip("'")  # Remove quotes
                                
                                # Set environment variable
                                os.environ[key] = value
                                
                                if key == 'SLACK_WEBHOOK_URL':
                                    print(f"📋 Found SLACK_WEBHOOK_URL in {env_path}")
                                    print(f"   Value: {value[:50]}...")
                    
                    break  # Stop after finding first valid .env file
                    
                except Exception as e:
                    print(f"❌ Error reading {env_path}: {e}")
                    continue
        
        if not env_found:
            print("❌ No .env file found in any expected location")
            print("📁 Expected locations:")
            for loc in possible_locations:
                print(f"   - {Path(loc).absolute()}")
        
        # Try to get the webhook URL from environment
        self.webhook_url = os.getenv('SLACK_WEBHOOK_URL')
        if not self.webhook_url:
            # Try alternative environment variable names
            self.webhook_url = os.getenv('SLACK_WEBHOOK') or os.getenv('WEBHOOK_URL')
        
        # Also try loading with python-dotenv if available
        try:
            from dotenv import load_dotenv
            for location in possible_locations:
                if Path(location).exists():
                    print(f"🔄 Trying dotenv load from: {Path(location).absolute()}")
                    success = load_dotenv(location, override=True)
                    print(f"   Dotenv load result: {success}")
                    if success:
                        new_webhook = os.getenv('SLACK_WEBHOOK_URL')
                        if new_webhook and not self.webhook_url:
                            self.webhook_url = new_webhook
                            print(f"✅ Got webhook URL from dotenv: {new_webhook[:50]}...")
                        break
        except ImportError:
            print("⚠️ python-dotenv not available, using manual parsing only")
        
        # Set enabled status
        self.enabled = bool(self.webhook_url)
        
        print(f"🎯 Final result: Webhook {'found' if self.enabled else 'not found'}")
        if self.enabled:
            print(f"   Using webhook: {self.webhook_url[:50]}...")
    
    def send_notification(self, message: str, channel: str = None, username: str = "QuantEdge Bot", icon_emoji: str = ":chart_with_upwards_trend:") -> bool:
        """Send a notification to Slack with detailed error handling"""
        if not self.enabled:
            print("❌ Slack not enabled - webhook URL missing")
            return False
        
        try:
            print(f"📤 Attempting to send Slack notification...")
            print(f"   Webhook URL: {self.webhook_url[:50]}...")
            print(f"   Message length: {len(message)} chars")
            
            payload = {
                "text": message,
                "username": username,
                "icon_emoji": icon_emoji
            }
            
            if channel:
                payload["channel"] = channel
            
            response = requests.post(
                self.webhook_url,
                json=payload,
                timeout=10
            )
            
            print(f"   Response status: {response.status_code}")
            print(f"   Response content: {response.text}")
            
            if response.status_code == 200:
                logger.info("✅ Slack notification sent successfully")
                print("✅ Slack notification sent successfully")
                return True
            else:
                logger.error(f"❌ Slack notification failed: {response.status_code}")
                print(f"❌ Slack notification failed: {response.status_code}")
                print(f"   Response: {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error sending Slack notification: {e}")
            print(f"❌ Error sending Slack notification: {e}")
            return False
    
    def notify_trade_executed(self, trade_data: Dict) -> bool:
        """Send notification when a trade is executed"""
        action_emoji = "🟢" if trade_data['action'] == 'BUY' else "🔴"
        
        message = f"""
{action_emoji} **TRADE EXECUTED** {action_emoji}

**{trade_data['action']} {trade_data['quantity']} {trade_data['symbol']}**
• Price: ${trade_data.get('price', 0):.2f}
• Strategy: {trade_data.get('strategy', 'MANUAL')}
• Confidence: {trade_data.get('confidence', 95):.1f}%
• Time: {datetime.now().strftime('%H:%M:%S PST')}

💰 **Value: ${trade_data.get('price', 0) * trade_data['quantity']:,.2f}**
        """
        
        return self.send_notification(
            message, 
            username="QuantEdge Trader",
            icon_emoji=":money_with_wings:"
        )
    
    def notify_auto_trading_started(self, config: Dict) -> bool:
        """Notify when auto trading starts"""
        enabled_strategies = [k for k, v in config.get('strategies_enabled', {}).items() if v]
        
        message = f"""
🚀 **AUTO TRADING STARTED** 🚀

**Configuration:**
• Strategies: {', '.join(enabled_strategies)}
• Max Positions: {config.get('max_positions', 5)}
• Min Confidence: {config.get('min_confidence', 65)}%
• Scan Frequency: {config.get('scan_frequency_minutes', 5)} minutes
• Daily Trade Limit: {config.get('max_daily_trades', 10)}

🎯 **Ready to trade during market hours!**
⏰ Started at: {datetime.now().strftime('%H:%M:%S PST')}
        """
        
        return self.send_notification(
            message,
            username="QuantEdge System",
            icon_emoji=":rocket:"
        )
    
    def notify_auto_trading_stopped(self) -> bool:
        """Notify when auto trading stops"""
        message = f"""
⏹️ **AUTO TRADING STOPPED** ⏹️

Auto trading has been disabled.
⏰ Stopped at: {datetime.now().strftime('%H:%M:%S PST')}

Manual trading is still available.
        """
        
        return self.send_notification(
            message,
            username="QuantEdge System", 
            icon_emoji=":stop_sign:"
        )
    
    def notify_market_scan(self, scan_results: Dict) -> bool:
        """Notify about market scan results"""
        if scan_results.get('signals_found', 0) == 0:
            return False  # Don't spam for no signals
        
        message = f"""
🔍 **MARKET SCAN COMPLETED** 🔍

• Signals Found: {scan_results.get('signals_found', 0)}
• Trades Executed: {scan_results.get('trades_executed', 0)}
• Current Positions: {scan_results.get('current_positions', 0)}
• Scan Time: {datetime.now().strftime('%H:%M:%S PST')}

{scan_results.get('summary', 'Scan completed successfully')}
        """
        
        return self.send_notification(
            message,
            username="QuantEdge Scanner",
            icon_emoji=":mag:"
        )
    
    def notify_position_update(self, position_data: Dict) -> bool:
        """Notify about position updates"""
        pnl_emoji = "🟢" if position_data.get('unrealized_pl', 0) >= 0 else "🔴"
        
        message = f"""
📊 **POSITION UPDATE** 📊

**{position_data['symbol']}**
• Quantity: {position_data.get('qty', 0)}
• Entry Price: ${position_data.get('avg_entry_price', 0):.2f}
• Current Price: ${position_data.get('current_price', 0):.2f}
• Market Value: ${position_data.get('market_value', 0):,.2f}

{pnl_emoji} **P&L: ${position_data.get('unrealized_pl', 0):+,.2f} ({position_data.get('unrealized_plpc', 0):+.2f}%)**
        """
        
        return self.send_notification(
            message,
            username="QuantEdge Portfolio",
            icon_emoji=":bar_chart:"
        )
    
    def notify_daily_summary(self, summary: Dict) -> bool:
        """Send daily trading summary"""
        message = f"""
📈 **DAILY TRADING SUMMARY** 📈

**Today's Performance:**
• Trades Executed: {summary.get('trades_executed', 0)}
• Winning Trades: {summary.get('winning_trades', 0)}
• Losing Trades: {summary.get('losing_trades', 0)}
• Win Rate: {summary.get('win_rate', 0):.1f}%

💰 **P&L: ${summary.get('total_pnl', 0):+,.2f}**
📊 Portfolio Value: ${summary.get('portfolio_value', 0):,.2f}

⏰ Summary for: {datetime.now().strftime('%Y-%m-%d')}
        """
        
        return self.send_notification(
            message,
            username="QuantEdge Daily Report",
            icon_emoji=":calendar:"
        )
    
    def notify_error(self, error_message: str, component: str = "System") -> bool:
        """Notify about system errors"""
        message = f"""
⚠️ **SYSTEM ERROR** ⚠️

**Component:** {component}
**Error:** {error_message}
**Time:** {datetime.now().strftime('%H:%M:%S PST')}

Please check the system and logs.
        """
        
        return self.send_notification(
            message,
            username="QuantEdge Alert",
            icon_emoji=":warning:"
        )
    
    def notify_connection_status(self, status: str, details: str = "") -> bool:
        """Notify about connection status changes"""
        emoji = "🟢" if status == "CONNECTED" else "🔴"
        
        message = f"""
{emoji} **CONNECTION STATUS: {status}** {emoji}

{details}
⏰ Time: {datetime.now().strftime('%H:%M:%S PST')}
        """
        
        return self.send_notification(
            message,
            username="QuantEdge Connection",
            icon_emoji=":satellite:"
        )
    
    def test_notification(self) -> bool:
        """Test Slack notification system with detailed debugging"""
        print("🧪 Testing Slack notification system...")
        
        message = f"""
🧪 **QUANTEDGE NOTIFICATION TEST** 🧪

This is a test notification from your QuantEdge trading platform.

✅ Slack integration is working properly!
🚀 Auto trading platform is ready
⏰ Test sent at: {datetime.now().strftime('%H:%M:%S PST')}

Happy trading! 💰
        """
        
        result = self.send_notification(
            message,
            username="QuantEdge Test",
            icon_emoji=":white_check_mark:"
        )
        
        print(f"🧪 Test result: {'Success' if result else 'Failed'}")
        return result

# Manual environment setup as fallback
print("=" * 60)
print("🔍 SLACK NOTIFICATIONS - MANUAL ENVIRONMENT SETUP")
print("=" * 60)

# Manually set known credentials as fallback
if not os.getenv('SLACK_WEBHOOK_URL'):
    print("🔧 Setting up known Slack webhook URL...")
    os.environ['SLACK_WEBHOOK_URL'] = 'https://hooks.slack.com/services/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'
    print("✅ Slack webhook URL set manually")

if not os.getenv('ALPACA_API_KEY'):
    print("🔧 Setting up known Alpaca credentials...")
    os.environ['ALPACA_API_KEY'] = 'PKO5L1R7M5OQQQAOD5E8'
    os.environ['ALPACA_SECRET_KEY'] = 'R9Z8aYjasLPG5CEZ1OuJA8t6F5rxqRSq2Qrs2jgf'
    os.environ['ALPACA_BASE_URL'] = 'https://paper-api.alpaca.markets'
    print("✅ Alpaca credentials set manually")

print("=" * 60)

# Global instance for easy access
slack_notifier = SlackNotificationSystem()

def send_slack_notification(message: str, **kwargs) -> bool:
    """Convenience function to send Slack notification"""
    return slack_notifier.send_notification(message, **kwargs)

def notify_trade(trade_data: Dict) -> bool:
    """Convenience function to notify about trades"""
    return slack_notifier.notify_trade_executed(trade_data)