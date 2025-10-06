# QuantEdge Complete Trading Platform

A comprehensive automated trading platform with Streamlit dashboard, auto trading capabilities, and Slack integration.

## 🚀 Quick Start

1. **Install Dependencies:**
   ```bash
   pip install streamlit pandas numpy plotly python-dotenv requests schedule threading
   ```

2. **Set up Environment Variables** (create `.env` in dashboard folder):
   ```
   ALPACA_API_KEY=your_alpaca_api_key
   ALPACA_SECRET_KEY=your_alpaca_secret_key
   ALPACA_BASE_URL=https://paper-api.alpaca.markets  # For paper trading
   SLACK_WEBHOOK_URL=your_slack_webhook_url
   ```

3. **Launch Dashboard:**
   ```bash
   cd dashboard
   streamlit run quantedge_complete_platform.py --server.port 8501
   ```

## 📁 File Structure

```
dashboard/
├── quantedge_complete_platform.py    # Main Streamlit dashboard
├── enhanced_auto_trader_slack.py     # Enhanced trading engine
├── continuous_auto_trader.py         # Background trading automation
├── slack_focused_alerts.py           # Slack notification system
├── auto_trader_config.json          # Configuration settings
├── quantedge_auto_trading.log       # Trading logs
├── __init__.py                      # Package initialization
├── .env                             # Environment variables (create this)
└── README.md                        # This file
```

## 🌟 Features

- **151+ Symbol Universe**: Comprehensive stock coverage across all sectors
- **Real-time Auto Trading**: Automated execution with configurable frequency
- **Dynamic Filter System**: Real-time configuration impact analysis
- **Slack Integration**: Instant trade alerts and notifications
- **Risk Management**: Position sizing, stop losses, profit targets
- **Portfolio Management**: Real-time portfolio tracking and analysis

## 🎯 Usage

1. **Dashboard**: Access the complete trading platform at `http://localhost:8501`
2. **Configuration**: Customize trading parameters in the Configuration tab
3. **Auto Trading**: Enable/disable automated trading with real frequency controls
4. **Monitoring**: Track performance in Command Center and Portfolio tabs

## 🛡️ Safety

- Paper trading mode enabled by default
- All real money trading requires explicit confirmation
- Comprehensive risk management controls
- Position limits and allocation controls

## 📊 System Status

- ✅ Enhanced Trader: Fully functional
- ✅ Auto Trading: Working frequency controls
- ✅ Dynamic Summaries: Fixed and rendering properly
- ✅ Slack Alerts: Integrated notification system