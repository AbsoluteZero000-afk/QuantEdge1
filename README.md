# 🚀 QuantEdge Professional Trading Suite

**A comprehensive algorithmic trading system with signal filtering, real Alpaca execution, and Slack notifications.**

## ✨ Features

### 🎯 **Signal Filtering & Analysis**
- **Professional signal filtering** by strategy, confidence, and trade count
- **8-factor quantitative analysis** with momentum, RSI, and volume indicators
- **Strategy classification**: MOMENTUM, BREAKOUT, TREND_FOLLOWING, MEAN_REVERSION
- **Risk-managed position sizing** with portfolio allocation controls

### 🚀 **Real Trading Execution**
- **Real Alpaca API integration** with actual order placement
- **Paper and live trading** modes with safety controls
- **Professional trade attribution** with strategy and confidence tracking
- **Complete audit trail** with order IDs and execution details

### 📊 **Professional Analytics**
- **Enhanced trade journal** with detailed performance analytics
- **Portfolio diversification analysis** with correlation metrics
- **Strategy performance tracking** with confidence vs P&L analysis
- **Export capabilities** for advanced analysis

### 🔔 **Real-time Notifications**
- **Slack integration** with detailed trade notifications
- **Live order status** updates with real Alpaca order IDs
- **System health monitoring** and error alerts
- **Market session** start/end notifications

### 🏆 **Professional Dashboard**
- **Streamlit-based interface** with institutional-grade UI
- **Signal filtering controls** with real-time preview
- **Position sizing calculator** with portfolio allocation
- **Execution confirmation** with safety controls

## 🏗️ Architecture

quantedge/
├── dashboard/ # Main Streamlit dashboard
├── trader/ # Real Alpaca trading engine
├── alerts/ # Slack notification system
├── journal/ # Trade logging and analytics
├── analytics/ # Portfolio and performance analysis
├── monitoring/ # System health and performance
├── strategies/ # Trading strategy implementations
├── data_ingestion/ # Market data loading
├── database/ # Database interface and management
├── risk_manager/ # Risk management and position sizing
└── backtester/ # Strategy backtesting framework
text

## 🚀 Quick Start

### Prerequisites
pip install streamlit pandas numpy plotly alpaca-trade-api sqlalchemy python-dotenv structlog requests
text

### Configuration
1. Copy `.env.example` to `.env`
2. Add your credentials:
Alpaca Trading
ALPACA_API_KEY=your_alpaca_key
ALPACA_SECRET_KEY=your_alpaca_secret
ALPACA_BASE_URL=https://paper-api.alpaca.markets
Database
DATABASE_URL=postgresql://user:pass@localhost/quantedge
Slack Notifications
SLACK_WEBHOOK_URL=your_slack_webhook
Market Data
FMP_API_KEY=your_fmp_key
text

### Launch System
Start enhanced analytics (Terminal 1)
streamlit run professional_trade_journal.py --server.port 8502
Start main dashboard (Terminal 2)
streamlit run dashboard/app.py --server.port 8501
text

### Access
- **Main Dashboard**: http://localhost:8501
- **Enhanced Analytics**: http://localhost:8502

## 🎯 Usage

### 1. Signal Filtering
- Select trading strategies (MOMENTUM, BREAKOUT, etc.)
- Set confidence thresholds (60-100%)
- Choose professional grades (STANDARD, PROFESSIONAL, INSTITUTIONAL)
- Limit number of trades (1-8)

### 2. Position Sizing
- Configure portfolio value
- Set risk per trade (5-25%)
- Review calculated positions and total investment
- Verify portfolio allocation percentages

### 3. Execution
- Choose Paper Trading (safe testing) or Live Trading
- Review execution summary and safety controls
- Execute filtered trades through real Alpaca API
- Monitor Slack notifications and trade journal

### 4. Analytics
- Access enhanced analytics at port 8502
- Review strategy performance and P&L analysis
- Export trade data for advanced analysis
- Monitor portfolio diversification and risk metrics

## 📊 Key Components

### Enhanced Auto Trader
- **Real Alpaca order placement** with `submit_order()` API calls
- **Comprehensive error handling** and retry logic
- **Position sizing** with risk management
- **Complete trade attribution** with metadata

### Signal Intelligence  
- **8-factor analysis** combining momentum, volume, and technical indicators
- **Strategy classification** with confidence scoring
- **Professional grading** system (STANDARD/PROFESSIONAL/INSTITUTIONAL)
- **Risk level assessment** (LOW/MODERATE/HIGH)

### Professional Dashboard
- **Institutional-grade UI** with advanced filtering
- **Real-time signal preview** with position sizing
- **Safety controls** for live trading
- **Complete execution workflow** with confirmations

## 🛡️ Safety Features

- **Paper trading mode** for safe testing
- **Live trading confirmations** with explicit authorization
- **Position size limits** with portfolio allocation caps
- **Error handling** with graceful fallbacks
- **Complete audit trail** with Slack and journal logging

## 🔧 System Requirements

- Python 3.8+
- PostgreSQL database
- Alpaca brokerage account
- Slack workspace (optional)
- Market data provider (FMP)

## 📈 Performance

- **Real-time signal generation** with < 5 second execution
- **Scalable to 100+ symbols** with optimized database queries
- **Professional-grade backtesting** with Monte Carlo analysis
- **Complete system monitoring** with health checks

## 🤝 Contributing

This is a professional trading system. Please ensure any contributions:
- Follow existing code patterns and documentation
- Include comprehensive error handling
- Maintain security best practices
- Add appropriate tests and logging

## ⚠️ Disclaimer

This software is for educational and research purposes. Trading involves significant risk of loss. Always test thoroughly with paper trading before using real capital. The authors are not responsible for any financial losses incurred through the use of this software.

## 📄 License

MIT License - see LICENSE file for details.

---

**Built with ❤️ for systematic trading and quantitative finance**
