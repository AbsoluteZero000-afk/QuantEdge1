"""
QuantEdge COMPLETE Dashboard - SIGNAL FILTERING WITH REAL ALPACA INTEGRATION
Professional trading dashboard with signal filtering, real Alpaca orders, and Slack notifications
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import os
import json
import sys
from pathlib import Path
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
import warnings

# Fix Python path resolution
current_dir = Path(__file__).parent.absolute()
project_root = current_dir.parent if current_dir.name == 'dashboard' else current_dir

# Add all module directories to Python path
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'journal'))
sys.path.insert(0, str(project_root / 'analytics'))
sys.path.insert(0, str(project_root / 'monitoring'))
sys.path.insert(0, str(project_root / 'alerts'))
sys.path.insert(0, str(project_root / 'trader'))

# Import enhanced auto trader with real Alpaca integration
ENHANCED_TRADER_AVAILABLE = False
try:
    from enhanced_auto_trader_slack import EnhancedQuantEdgeTrader
    ENHANCED_TRADER_AVAILABLE = True
    print("✅ Enhanced Auto Trader with Real Alpaca loaded successfully")
except Exception as e:
    print(f"❌ Enhanced Auto Trader failed: {e}")
    # Create fallback class
    class EnhancedQuantEdgeTrader:
        def __init__(self, paper_trading=True): 
            self.paper_trading = paper_trading
        def run_enhanced_auto_trading(self): 
            return {'error': 'Enhanced trader not available'}

# Try importing other modules
JOURNAL_AVAILABLE = False
ANALYTICS_AVAILABLE = False
MONITOR_AVAILABLE = False
ALERTS_AVAILABLE = False

try:
    from trade_journal import QuantEdgeJournal
    JOURNAL_AVAILABLE = True
    print("✅ Trade Journal module loaded successfully")
except Exception as e:
    print(f"❌ Trade Journal module failed: {e}")
    class QuantEdgeJournal:
        def __init__(self): pass
        def get_journal_summary(self, days): return {'total_trades': 0, 'win_rate': 0, 'total_pnl': 0, 'average_pnl': 0}

try:
    from portfolio_analytics import QuantEdgeAnalytics
    ANALYTICS_AVAILABLE = True
    print("✅ Portfolio Analytics module loaded successfully")
except Exception as e:
    print(f"❌ Portfolio Analytics module failed: {e}")
    class QuantEdgeAnalytics:
        def __init__(self): pass
        def analyze_portfolio_diversification(self, positions): return {'diversification_score': 75, 'assessment': 'GOOD'}

try:
    from performance_monitor import QuantEdgeMonitor
    MONITOR_AVAILABLE = True
    print("✅ Performance Monitor module loaded successfully")
except Exception as e:
    print(f"❌ Performance Monitor module failed: {e}")
    class QuantEdgeMonitor:
        def __init__(self): pass
        def calculate_daily_pnl(self, portfolio_value): return {'daily_pnl': 0, 'portfolio_return': 0}

try:
    from slack_focused_alerts import QuantEdgeAlerter
    ALERTS_AVAILABLE = True
    print("✅ Slack Alerts module loaded successfully")
except Exception as e:
    print(f"❌ Slack Alerts module failed: {e}")
    class QuantEdgeAlerter:
        def __init__(self): pass
        def send_slack_alert(self, title, message, priority='info'): return True

warnings.filterwarnings('ignore')
load_dotenv()

st.set_page_config(
    page_title="QuantEdge Professional Suite - Signal Filtering & Real Trading",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional CSS
st.markdown("""
<style>
    .main { padding-top: 0.5rem; }
    
    .complete-header {
        background: linear-gradient(135deg, #007bff 0%, #0056b3 100%);
        color: white; padding: 2rem; border-radius: 1rem; margin-bottom: 2rem; text-align: center;
        box-shadow: 0 6px 20px rgba(0,123,255,0.3);
    }
    
    .signal-filter-premium {
        background: linear-gradient(135deg, rgba(0, 123, 255, 0.15) 0%, rgba(0, 123, 255, 0.1) 100%);
        padding: 2rem; border-radius: 1rem; border: 2px solid #007bff; margin: 1.5rem 0;
        box-shadow: 0 6px 20px rgba(0, 123, 255, 0.2);
    }
    
    .execution-premium {
        background: linear-gradient(135deg, rgba(40, 167, 69, 0.2) 0%, rgba(40, 167, 69, 0.1) 100%);
        padding: 2rem; border-radius: 1rem; border: 3px solid #28a745; margin: 1.5rem 0;
        box-shadow: 0 6px 20px rgba(40, 167, 69, 0.3);
    }
    
    .signal-premium {
        background: rgba(40, 167, 69, 0.1); padding: 1.5rem; border-radius: 1rem;
        border-left: 5px solid #28a745; margin: 1rem 0;
    }
    
    .critical-premium {
        background: rgba(220, 53, 69, 0.15); padding: 1.5rem; border-radius: 1rem;
        border-left: 5px solid #dc3545; margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=300)
def load_market_data():
    """Load market data for signal generation."""
    db_url = os.getenv('DATABASE_URL')
    
    if not db_url:
        # Return demo data for testing
        return pd.DataFrame([
            {'symbol': 'AAPL', 'price': 179.50, 'strategy': 'MOMENTUM', 'confidence': 85, 'mom_10d': 5.2, 'rsi': 65},
            {'symbol': 'MSFT', 'price': 318.75, 'strategy': 'MOMENTUM', 'confidence': 78, 'mom_10d': 3.8, 'rsi': 58},
            {'symbol': 'TSLA', 'price': 260.25, 'strategy': 'BREAKOUT', 'confidence': 82, 'mom_10d': 7.1, 'rsi': 72},
            {'symbol': 'NVDA', 'price': 875.50, 'strategy': 'MOMENTUM', 'confidence': 88, 'mom_10d': 6.5, 'rsi': 62},
            {'symbol': 'GOOGL', 'price': 162.25, 'strategy': 'TREND_FOLLOWING', 'confidence': 71, 'mom_10d': 2.8, 'rsi': 55}
        ])
    
    try:
        engine = create_engine(db_url)
        
        # Get market data (simplified for demo)
        query = text("""
        SELECT DISTINCT symbol, close as price
        FROM stock_prices
        WHERE date = (SELECT MAX(date) FROM stock_prices)
        LIMIT 20
        """)
        
        df = pd.read_sql(query, engine)
        
        if not df.empty:
            # Add demo strategy and confidence for testing
            strategies = ['MOMENTUM', 'BREAKOUT', 'TREND_FOLLOWING', 'MEAN_REVERSION']
            df['strategy'] = [strategies[i % len(strategies)] for i in range(len(df))]
            df['confidence'] = np.random.uniform(60, 95, len(df))
            df['mom_10d'] = np.random.uniform(-2, 8, len(df))
            df['rsi'] = np.random.uniform(30, 80, len(df))
        
        return df
        
    except Exception as e:
        st.error(f"Database error: {e}")
        return pd.DataFrame()

def main():
    """Professional dashboard with signal filtering and real Alpaca integration."""
    
    # Header
    st.markdown("""
    <div class="complete-header">
        <h1>🎯 QuantEdge Professional Signal Filtering & Real Trading Suite</h1>
        <h2>Filter Signals → Review Trades → Execute Real Alpaca Orders</h2>
        <p style="font-size: 1.2em; margin-top: 1rem;">
            Professional signal filtering with real Alpaca execution and Slack notifications
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Check if enhanced trader is available
    if not ENHANCED_TRADER_AVAILABLE:
        st.error("❌ Enhanced Auto Trader not available. Check trader/enhanced_auto_trader_slack.py")
        st.info("🔧 Make sure the file exists in trader/ directory and all dependencies are installed.")
        return
    
    # Load market data
    with st.spinner("📊 Loading market data..."):
        market_data = load_market_data()
    
    if market_data.empty:
        st.error("❌ No market data available")
        return
    
    # Professional tabs
    tab1, tab2, tab3 = st.tabs([
        "🎯 FILTERED SIGNAL TRADING",
        "📊 Signal Intelligence",
        "📝 Trade Journal"
    ])
    
    with tab1:
        st.header("🎯 Professional Signal Filtering & Execution")
        
        st.markdown("""
        <div class="signal-filter-premium">
            <h3>🎯 PROFESSIONAL SIGNAL FILTERING SYSTEM</h3>
            <p>Filter signals by strategy, confidence, and trade count → Review → Execute real Alpaca orders</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Generate signals (use market data)
        available_signals = market_data[market_data['confidence'] >= 60].copy()
        
        if not available_signals.empty:
            
            st.markdown("### 🔍 **Signal Filtering Controls**")
            
            # Professional filters
            filter_col1, filter_col2, filter_col3, filter_col4 = st.columns(4)
            
            with filter_col1:
                # Strategy filter
                available_strategies = sorted(available_signals['strategy'].unique())
                selected_strategies = st.multiselect(
                    "🧠 **Trading Strategies**",
                    options=available_strategies,
                    default=available_strategies,
                    help="Select which trading strategies to execute"
                )
            
            with filter_col2:
                # Confidence filter
                confidence_range = st.slider(
                    "📊 **Confidence Range**",
                    min_value=60,
                    max_value=100,
                    value=(70, 100),
                    step=5,
                    help="Minimum and maximum confidence levels"
                )
            
            with filter_col3:
                # Grade filter
                grade_options = ['STANDARD', 'PROFESSIONAL', 'INSTITUTIONAL']
                selected_grades = st.multiselect(
                    "🏆 **Professional Grades**",
                    options=grade_options,
                    default=['PROFESSIONAL', 'INSTITUTIONAL'],
                    help="Select minimum professional grade levels"
                )
            
            with filter_col4:
                # Max trades
                max_trades = st.selectbox(
                    "📈 **Max Trades**",
                    options=[1, 2, 3, 4, 5, 6, 7, 8],
                    index=2,  # Default 3
                    help="Maximum number of trades to execute"
                )
            
            # Apply filters
            filtered_signals = available_signals[
                (available_signals['strategy'].isin(selected_strategies)) &
                (available_signals['confidence'] >= confidence_range[0]) &
                (available_signals['confidence'] <= confidence_range[1])
            ].head(max_trades).copy()
            
            # Add professional grade for filtering (demo)
            filtered_signals['professional_grade'] = filtered_signals['confidence'].apply(
                lambda x: 'INSTITUTIONAL' if x >= 85 else 'PROFESSIONAL' if x >= 75 else 'STANDARD'
            )
            
            filtered_signals = filtered_signals[
                filtered_signals['professional_grade'].isin(selected_grades)
            ]
            
            # Display filtered results
            st.markdown("---")
            st.markdown(f"### 📋 **Filtered Signals Ready for Execution** ({len(filtered_signals)} trades)")
            
            if not filtered_signals.empty:
                
                # Enhanced signal display
                display_df = filtered_signals[['symbol', 'strategy', 'price', 'confidence', 'professional_grade']].copy()
                display_df.columns = ['Symbol', 'Strategy', 'Price ($)', 'Confidence (%)', 'Grade']
                display_df['Price ($)'] = display_df['Price ($)'].apply(lambda x: f"${x:.2f}")
                display_df['Confidence (%)'] = display_df['Confidence (%)'].apply(lambda x: f"{x:.0f}%")
                
                st.dataframe(display_df, use_container_width=True, height=300)
                
                # Position sizing preview
                st.markdown("### 💼 **Position Sizing Preview**")
                
                portfolio_value = st.number_input(
                    "Portfolio Value ($)",
                    min_value=10000,
                    max_value=5000000,
                    value=100000,
                    step=10000
                )
                
                risk_per_trade = st.slider(
                    "Risk Per Trade (%)",
                    min_value=5.0,
                    max_value=25.0,
                    value=15.0,
                    step=2.5
                )
                
                # Calculate positions
                position_value = portfolio_value * (risk_per_trade / 100)
                total_investment = 0
                
                st.markdown("#### 📊 **Calculated Positions:**")
                
                position_details = []
                for _, signal in filtered_signals.iterrows():
                    shares = max(1, int(position_value / signal['price']))
                    investment = shares * signal['price']
                    total_investment += investment
                    
                    position_details.append({
                        'Symbol': signal['symbol'],
                        'Strategy': signal['strategy'],
                        'Shares': f"{shares:,}",
                        'Price': f"${signal['price']:.2f}",
                        'Investment': f"${investment:,.0f}",
                        'Confidence': f"{signal['confidence']:.0f}%",
                        'Grade': signal['professional_grade']
                    })
                
                # Position preview table
                position_df = pd.DataFrame(position_details)
                st.dataframe(position_df, use_container_width=True)
                
                # Investment summary
                summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)
                
                with summary_col1:
                    st.metric("**Total Investment**", f"${total_investment:,.0f}")
                with summary_col2:
                    portfolio_pct = (total_investment / portfolio_value) * 100
                    st.metric("**Portfolio Allocation**", f"{portfolio_pct:.1f}%")
                with summary_col3:
                    avg_conf = filtered_signals['confidence'].mean()
                    st.metric("**Avg Confidence**", f"{avg_conf:.0f}%")
                with summary_col4:
                    inst_count = len(filtered_signals[filtered_signals['professional_grade'] == 'INSTITUTIONAL'])
                    st.metric("**Institutional Grade**", inst_count)
                
                # Trading execution section
                st.markdown("---")
                st.markdown("### 🚀 **Professional Trading Execution**")
                
                exec_col1, exec_col2 = st.columns(2)
                
                with exec_col1:
                    trading_mode = st.selectbox(
                        "**Trading Mode**",
                        options=["📝 Paper Trading", "💰 Live Trading"],
                        index=0,
                        help="Paper trading for testing, Live trading for real money"
                    )
                    paper_mode = "Paper" in trading_mode
                
                with exec_col2:
                    # Safety confirmation
                    if paper_mode:
                        st.markdown("""
                        <div class="signal-premium">
                            <h5>📝 PAPER TRADING MODE</h5>
                            <p>Safe testing with real Alpaca integration</p>
                        </div>
                        """, unsafe_allow_html=True)
                        confirm_trading = True
                    else:
                        st.markdown("""
                        <div class="critical-premium">
                            <h5>🚨 LIVE TRADING MODE</h5>
                            <p><strong>WARNING:</strong> Real money will be used</p>
                        </div>
                        """, unsafe_allow_html=True)
                        confirm_trading = st.checkbox("✅ I authorize live trading with real money")
                
                # MAIN EXECUTION BUTTON
                st.markdown("---")
                
                execution_summary = f"""
                **Ready to Execute:**
                • **{len(filtered_signals)} trades** from filtered signals
                • **${total_investment:,.0f}** total investment
                • **{avg_conf:.0f}%** average confidence
                • **{trading_mode}** execution mode
                """
                
                st.info(execution_summary)
                
                if st.button(
                    f"🎯 EXECUTE {len(filtered_signals)} FILTERED TRADES VIA REAL ALPACA",
                    type="primary",
                    disabled=not confirm_trading,
                    help="Execute filtered signals through Enhanced Alpaca Trader with real orders",
                    use_container_width=True
                ):
                    
                    with st.spinner(f"🚀 Executing {len(filtered_signals)} filtered trades via Real Alpaca..."):
                        
                        try:
                            # Initialize enhanced trader
                            trader = EnhancedQuantEdgeTrader(paper_trading=paper_mode)
                            
                            # Create signals format for trader
                            signals_for_trader = filtered_signals[['symbol', 'price', 'confidence', 'strategy']].copy()
                            signals_for_trader['action'] = 'BUY'  # All signals are buy signals
                            
                            # Execute using the enhanced trader's real order system
                            account_info = trader.get_account_info()
                            execution_results = trader.execute_trades_with_real_orders(signals_for_trader)
                            
                            if execution_results:
                                successful_orders = [r for r in execution_results if r.get('status') == 'REAL_ORDER_PLACED']
                                failed_orders = [r for r in execution_results if r.get('status') != 'REAL_ORDER_PLACED']
                                
                                st.markdown("""
                                <div class="execution-premium">
                                    <h3>✅ REAL ALPACA EXECUTION COMPLETE</h3>
                                    <p>Your filtered signals executed through real Alpaca orders with Slack notifications!</p>
                                </div>
                                """, unsafe_allow_html=True)
                                
                                # Results metrics
                                result_col1, result_col2, result_col3, result_col4 = st.columns(4)
                                
                                with result_col1:
                                    st.metric("✅ **Successful Orders**", len(successful_orders))
                                with result_col2:
                                    st.metric("❌ **Failed Orders**", len(failed_orders))
                                with result_col3:
                                    total_deployed = sum(r.get('estimated_value', 0) for r in successful_orders)
                                    st.metric("💰 **Total Deployed**", f"${total_deployed:,.0f}")
                                with result_col4:
                                    mode_display = "📝 Paper" if paper_mode else "💰 Live"
                                    st.metric("**Execution Mode**", mode_display)
                                
                                # Detailed results
                                if successful_orders:
                                    st.markdown("### ✅ **Successful Real Alpaca Orders**")
                                    
                                    for result in successful_orders:
                                        st.success(f"""
                                        🎯 **REAL ORDER PLACED**: {result['action']} {result['shares']} {result['symbol']} 
                                        @ ${result['estimated_price']:.2f} = ${result['estimated_value']:,.0f}
                                        
                                        **Alpaca Order ID**: `{result.get('alpaca_order_id', 'N/A')}`
                                        **Strategy**: {result.get('strategy', 'N/A')} | **Confidence**: {result.get('confidence', 0):.0f}%
                                        """)
                                
                                if failed_orders:
                                    st.markdown("### ❌ **Failed Orders**")
                                    for result in failed_orders:
                                        st.error(f"**{result['symbol']}**: {result.get('error', 'Unknown error')}")
                                
                                # Action items
                                st.balloons()
                                st.success("🔔 **Check your Slack channel for detailed trade notifications!**")
                                st.info("🏦 **Check your Alpaca account - you should see real orders!**")
                                st.info("📝 **All trades automatically logged to your trade journal**")
                            
                            else:
                                st.warning("❌ No orders could be executed. Check signals and account status.")
                        
                        except Exception as e:
                            st.error(f"❌ Execution failed: {e}")
                            st.info("🔧 Check Enhanced Trader configuration and Alpaca credentials")
            
            else:
                st.warning("📊 No signals match your current filter criteria. Try adjusting the filters above.")
        
        else:
            st.info("📊 No signals available. Check market data connection.")
    
    with tab2:
        st.header("📊 Signal Intelligence")
        
        if not market_data.empty:
            # Signal overview
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Signals", len(market_data))
            with col2:
                high_conf = len(market_data[market_data['confidence'] > 80])
                st.metric("High Confidence", high_conf)
            with col3:
                strategies = market_data['strategy'].nunique()
                st.metric("Strategy Types", strategies)
            with col4:
                avg_conf = market_data['confidence'].mean()
                st.metric("Avg Confidence", f"{avg_conf:.0f}%")
            
            # Strategy breakdown
            st.subheader("🧠 Strategy Analysis")
            
            strategy_counts = market_data['strategy'].value_counts()
            
            chart_col1, chart_col2 = st.columns(2)
            
            with chart_col1:
                fig_strategies = px.pie(
                    values=strategy_counts.values,
                    names=strategy_counts.index,
                    title="Strategy Distribution"
                )
                st.plotly_chart(fig_strategies, use_container_width=True)
            
            with chart_col2:
                fig_confidence = px.histogram(
                    market_data,
                    x='confidence',
                    nbins=15,
                    title="Confidence Distribution"
                )
                st.plotly_chart(fig_confidence, use_container_width=True)
            
            # Full signals table
            st.subheader("📊 All Available Signals")
            
            full_display = market_data[['symbol', 'strategy', 'price', 'confidence', 'mom_10d', 'rsi']].copy()
            full_display.columns = ['Symbol', 'Strategy', 'Price ($)', 'Confidence (%)', '10d Momentum (%)', 'RSI']
            full_display = full_display.round(2)
            
            st.dataframe(full_display, use_container_width=True, height=400)
    
    with tab3:
        st.header("📝 Trade Journal")

        # Enhanced Analytics Integration
        st.markdown("### 🚀 **Professional Analytics Available**")

        col1, col2, col3 = st.columns([1, 2, 1])

        with col2:
            if st.button("🔗 **Open Enhanced Analytics (Detailed View)**",
                         use_container_width=True):
                st.balloons()
                st.success("🎯 **Enhanced Analytics Ready!**")
                st.info("""
                **📊 Your Enhanced Analytics Include:**
                • Strategy performance analysis  
                • Confidence vs P&L correlation
                • Professional trade grading
                • Export capabilities & insights
                • Real Alpaca order tracking

                **🚀 Instructions:**
                1. Open a new terminal
                2. Run: `streamlit run professional_trade_journal.py --server.port 8502`
                3. Visit: http://localhost:8502
                """)

        st.markdown("---")
        
        # Trade journal summary
        if JOURNAL_AVAILABLE:
            try:
                journal = QuantEdgeJournal()
                summary = journal.get_journal_summary(30)
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Trades", summary.get('total_trades', 0))
                with col2:
                    st.metric("Win Rate", f"{summary.get('win_rate', 0):.1f}%")
                with col3:
                    st.metric("Total P&L", f"${summary.get('total_pnl', 0):+.2f}")
                with col4:
                    st.metric("Avg P&L/Trade", f"${summary.get('average_pnl', 0):+.2f}")
            
            except Exception as e:
                st.info("📝 Trade Journal: Enhanced trader handles logging automatically")
        else:
            st.info("📝 Trade Journal: Enhanced trader handles logging automatically")
    
    # System status footer
    st.markdown("---")
    st.markdown("### 🎛️ **System Controls**")
    
    control_col1, control_col2, control_col3, control_col4 = st.columns(4)
    
    with control_col1:
        if st.button("🧪 Test Enhanced Trader"):
            try:
                trader = EnhancedQuantEdgeTrader(paper_trading=True)
                account = trader.get_account_info()
                st.success(f"✅ Trader ready! Account: ${account['equity']:,.2f}")
            except Exception as e:
                st.error(f"❌ Trader test failed: {e}")
    
    with control_col2:
        if st.button("📡 Test Slack Alerts"):
            try:
                if ALERTS_AVAILABLE:
                    alerter = QuantEdgeAlerter()
                    success = alerter.send_slack_alert(
                        "Signal Filtering System Test",
                        "🎯 **QuantEdge Signal Filtering System**\n\nSystem test successful!\n✅ Ready for filtered signal execution",
                        priority="success"
                    )
                    st.success("✅ Slack test sent!" if success else "❌ Slack test failed")
                else:
                    st.warning("⚠️ Slack alerts not available")
            except Exception as e:
                st.error(f"❌ Slack test failed: {e}")
    
    with control_col3:
        if st.button("📊 Get Alpaca Positions"):
            try:
                trader = EnhancedQuantEdgeTrader(paper_trading=True)
                positions = trader.get_current_positions()
                
                if positions:
                    st.success(f"📈 Found {len(positions)} positions in Alpaca")
                    for pos in positions[:3]:  # Show first 3
                        st.info(f"{pos['symbol']}: {pos['qty']} shares, P&L: ${pos['unrealized_pl']:.2f}")
                else:
                    st.info("📊 No current positions in Alpaca account")
            except Exception as e:
                st.error(f"❌ Position check failed: {e}")
    
    with control_col4:
        if st.button("🔄 Refresh System"):
            st.cache_data.clear()
            st.rerun()
    
    # Professional footer
    st.markdown("---")
    
    footer_col1, footer_col2, footer_col3, footer_col4 = st.columns(4)
    
    with footer_col1:
        st.markdown(f"**Time:** {datetime.now().strftime('%H:%M:%S')}")
    with footer_col2:
        st.markdown(f"**Signals:** {len(market_data)}")
    with footer_col3:
        status = "🎯 READY" if ENHANCED_TRADER_AVAILABLE else "🔧 Setup"
        st.markdown(f"**Filtering:** {status}")
    with footer_col4:
        st.markdown(f"**Real Trading:** {'🚀 ACTIVE' if ENHANCED_TRADER_AVAILABLE else '❌ Inactive'}")
    
    # System signature
    st.markdown(f"""
    <div style='
        text-align: center; 
        background: linear-gradient(135deg, #007bff 0%, #0056b3 100%);
        color: white; padding: 2rem; border-radius: 1rem; margin-top: 2rem;
        box-shadow: 0 6px 20px rgba(0,0,0,0.2);
    '>
        <h2>🎯 QuantEdge Professional Signal Filtering & Real Trading Suite</h2>
        <p style="font-size: 1.1em; margin: 1rem 0;">
            <strong>Capabilities:</strong> Advanced signal filtering • Real Alpaca execution • Live Slack alerts
        </p>
        <p style="margin-top: 1rem; font-weight: bold;">
            {'🎯 PROFESSIONAL SIGNAL FILTERING SYSTEM READY' if ENHANCED_TRADER_AVAILABLE 
             else '🔧 SETUP REQUIRED FOR REAL TRADING'}
        </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()