"""
QuantEdge Professional Trade Journal - Optimized for Your Complete Schema
Takes full advantage of your comprehensive trading database
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import sqlite3
from pathlib import Path
import json
import numpy as np

def render_professional_trade_journal():
    """Render professional trade journal optimized for your complete schema."""
    
    st.markdown("# 📝 **QuantEdge Professional Trade Journal**")
    st.markdown("*Institutional-grade trade attribution, analysis, and learning system*")
    
    # Database connection
    journal_db_path = Path("data/trade_journal.db")
    
    if not journal_db_path.exists():
        st.warning("📊 Trade journal database not found.")
        st.info("Execute some trades to see comprehensive professional analytics!")
        return
    
    try:
        conn = sqlite3.connect(str(journal_db_path))
        trades_df = pd.read_sql_query("SELECT * FROM trades ORDER BY timestamp DESC", conn)
        conn.close()
        
    except Exception as e:
        st.error(f"❌ Database error: {e}")
        return
    
    if trades_df.empty:
        st.info("📝 No trades recorded yet. Execute your first trade to see professional analytics!")
        return
    
    # Convert datetime columns
    trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp'])
    trades_df['entry_date'] = pd.to_datetime(trades_df['entry_date'])
    trades_df['exit_date'] = pd.to_datetime(trades_df['exit_date'])
    trades_df['date'] = trades_df['timestamp'].dt.date
    
    # Professional Dashboard Header
    st.markdown("## 🏆 **Professional Trading Dashboard**")
    
    # Key Performance Metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_trades = len(trades_df)
        st.metric("**Total Trades**", total_trades)
    
    with col2:
        open_positions = len(trades_df[trades_df['exit_date'].isna()])
        st.metric("**Open Positions**", open_positions)
    
    with col3:
        avg_confidence = trades_df['confidence_level'].mean()
        st.metric("**Avg Confidence**", f"{avg_confidence:.1f}%")
    
    with col4:
        closed_trades = trades_df[trades_df['pnl_dollars'].notna()]
        if not closed_trades.empty:
            total_pnl = closed_trades['pnl_dollars'].sum()
            st.metric("**Total P&L**", f"${total_pnl:.2f}")
        else:
            st.metric("**Total P&L**", "Pending")
    
    # Professional Filters
    st.markdown("---")
    st.markdown("## 🔍 **Professional Analysis Filters**")
    
    filter_col1, filter_col2, filter_col3, filter_col4 = st.columns(4)
    
    with filter_col1:
        symbols_filter = st.multiselect(
            "📈 **Symbols**",
            options=sorted(trades_df['symbol'].unique()),
            default=[]
        )
    
    with filter_col2:
        strategies_filter = st.multiselect(
            "🧠 **Strategies**",
            options=trades_df['strategy'].dropna().unique(),
            default=list(trades_df['strategy'].dropna().unique())
        )
    
    with filter_col3:
        status_filter = st.selectbox(
            "📊 **Position Status**",
            options=['All', 'Open', 'Closed'],
            index=0
        )
    
    with filter_col4:
        confidence_range = st.slider(
            "🎯 **Confidence Range**",
            min_value=0,
            max_value=100,
            value=(60, 100),
            step=5
        )
    
    # Apply filters
    filtered_trades = trades_df.copy()
    
    if symbols_filter:
        filtered_trades = filtered_trades[filtered_trades['symbol'].isin(symbols_filter)]
    
    if strategies_filter:
        filtered_trades = filtered_trades[filtered_trades['strategy'].isin(strategies_filter)]
    
    if status_filter == 'Open':
        filtered_trades = filtered_trades[filtered_trades['exit_date'].isna()]
    elif status_filter == 'Closed':
        filtered_trades = filtered_trades[filtered_trades['exit_date'].notna()]
    
    # Confidence filter
    filtered_trades = filtered_trades[
        (filtered_trades['confidence_level'] >= confidence_range[0]) & 
        (filtered_trades['confidence_level'] <= confidence_range[1])
    ]
    
    st.markdown(f"### 📋 **Trade Analysis** ({len(filtered_trades)} trades)")
    
    if not filtered_trades.empty:
        
        # Enhanced Professional Trade Table
        display_trades = filtered_trades.copy()
        
        # Format for professional display
        display_trades['Entry Time'] = display_trades['entry_date'].dt.strftime('%m/%d %H:%M')
        display_trades['Symbol'] = display_trades['symbol']
        display_trades['Action'] = display_trades['action'].apply(
            lambda x: f"🟢 {x}" if x == 'BUY' else f"🔴 {x}"
        )
        display_trades['Shares'] = display_trades['shares']
        display_trades['Entry Price'] = display_trades['entry_price'].apply(lambda x: f"${x:.2f}")
        display_trades['Exit Price'] = display_trades['exit_price'].apply(
            lambda x: f"${x:.2f}" if pd.notna(x) else "Open"
        )
        display_trades['Strategy'] = display_trades['strategy']
        display_trades['Confidence'] = display_trades['confidence_level'].apply(lambda x: f"{x:.0f}%")
        display_trades['P&L'] = display_trades['pnl_dollars'].apply(
            lambda x: f"${x:+.2f}" if pd.notna(x) else "Pending"
        )
        display_trades['Grade'] = display_trades['trade_grade']
        display_trades['Mode'] = display_trades['execution_mode'].str.title()
        
        # Professional columns for display
        professional_columns = [
            'Entry Time', 'Symbol', 'Action', 'Shares', 'Entry Price', 
            'Exit Price', 'Strategy', 'Confidence', 'P&L', 'Grade', 'Mode'
        ]
        
        st.dataframe(
            display_trades[professional_columns],
            use_container_width=True,
            height=400
        )
        
        # Professional Analytics Charts
        st.markdown("---")
        st.markdown("## 📊 **Professional Analytics Dashboard**")
        
        chart_row1_col1, chart_row1_col2 = st.columns(2)
        
        with chart_row1_col1:
            # Strategy Performance
            strategy_analysis = filtered_trades.groupby('strategy').agg({
                'id': 'count',
                'confidence_level': 'mean',
                'pnl_dollars': 'sum'
            }).round(2)
            strategy_analysis.columns = ['Trades', 'Avg Confidence', 'Total P&L']
            
            if not strategy_analysis.empty:
                fig_strategy = px.bar(
                    strategy_analysis.reset_index(),
                    x='strategy',
                    y='Trades',
                    title="📈 Strategy Usage Analysis",
                    color='Avg Confidence',
                    color_continuous_scale='viridis'
                )
                fig_strategy.update_layout(height=300)
                st.plotly_chart(fig_strategy, use_container_width=True)
        
        with chart_row1_col2:
            # Confidence Distribution
            fig_confidence = px.histogram(
                filtered_trades,
                x='confidence_level',
                nbins=20,
                title="🎯 Confidence Level Distribution"
            )
            fig_confidence.update_layout(height=300)
            st.plotly_chart(fig_confidence, use_container_width=True)
        
        chart_row2_col1, chart_row2_col2 = st.columns(2)
        
        with chart_row2_col1:
            # P&L Analysis (for closed trades)
            closed_trades = filtered_trades[filtered_trades['pnl_dollars'].notna()]
            if not closed_trades.empty:
                fig_pnl = px.scatter(
                    closed_trades,
                    x='confidence_level',
                    y='pnl_dollars',
                    color='strategy',
                    size='shares',
                    title="💰 P&L vs Confidence Analysis",
                    hover_data=['symbol']
                )
                fig_pnl.update_layout(height=300)
                st.plotly_chart(fig_pnl, use_container_width=True)
            else:
                st.info("📊 P&L analysis available after closing positions")
        
        with chart_row2_col2:
            # Trading Activity Timeline
            daily_activity = filtered_trades.groupby('date').size().reset_index(name='trades')
            
            fig_timeline = px.line(
                daily_activity,
                x='date',
                y='trades',
                title="📅 Trading Activity Timeline",
                markers=True
            )
            fig_timeline.update_layout(height=300)
            st.plotly_chart(fig_timeline, use_container_width=True)
        
        # Professional Performance Analysis
        st.markdown("---")
        st.markdown("## 🧠 **Professional Performance Analysis**")
        
        analysis_col1, analysis_col2 = st.columns(2)
        
        with analysis_col1:
            st.markdown("#### 📊 **Strategy Breakdown**")
            strategy_summary = filtered_trades.groupby('strategy').agg({
                'id': 'count',
                'confidence_level': 'mean',
                'shares': 'sum'
            }).round(2)
            strategy_summary.columns = ['Trades', 'Avg Confidence (%)', 'Total Shares']
            st.dataframe(strategy_summary, use_container_width=True)
        
        with analysis_col2:
            st.markdown("#### 🎯 **Symbol Analysis**")
            symbol_summary = filtered_trades.groupby('symbol').agg({
                'id': 'count',
                'entry_price': 'mean',
                'confidence_level': 'mean'
            }).round(2)
            symbol_summary.columns = ['Trades', 'Avg Entry ($)', 'Avg Confidence (%)']
            st.dataframe(symbol_summary, use_container_width=True)
        
        # Professional Insights
        st.markdown("---")
        st.markdown("### 🎖️ **Professional Trading Insights**")
        
        insights_col1, insights_col2, insights_col3 = st.columns(3)
        
        with insights_col1:
            st.markdown("#### 🏆 **Best Performance**")
            
            if not filtered_trades.empty:
                top_strategy = filtered_trades['strategy'].value_counts().index[0]
                top_strategy_count = filtered_trades['strategy'].value_counts().iloc[0]
                st.success(f"**Most Used Strategy**: {top_strategy} ({top_strategy_count} trades)")
                
                highest_confidence = filtered_trades['confidence_level'].max()
                st.info(f"**Highest Confidence**: {highest_confidence:.0f}%")
        
        with insights_col2:
            st.markdown("#### 🎯 **Trading Discipline**")
            
            paper_trades = (filtered_trades['execution_mode'] == 'paper').sum()
            total = len(filtered_trades)
            st.info(f"**Paper Trading**: {paper_trades}/{total} ({paper_trades/total*100:.1f}%)")
            
            avg_confidence = filtered_trades['confidence_level'].mean()
            if avg_confidence > 75:
                st.success(f"**Confidence**: {avg_confidence:.1f}% - Excellent!")
            elif avg_confidence > 65:
                st.info(f"**Confidence**: {avg_confidence:.1f}% - Good")
            else:
                st.warning(f"**Confidence**: {avg_confidence:.1f}% - Consider higher thresholds")
        
        with insights_col3:
            st.markdown("#### 🚀 **Recommendations**")
            
            closed_trades = filtered_trades[filtered_trades['pnl_dollars'].notna()]
            if not closed_trades.empty:
                profitable = (closed_trades['pnl_dollars'] > 0).sum()
                total_closed = len(closed_trades)
                win_rate = profitable / total_closed
                
                if win_rate > 0.6:
                    st.success(f"🎉 **Win Rate**: {win_rate:.1%} - Excellent!")
                elif win_rate > 0.5:
                    st.info(f"📊 **Win Rate**: {win_rate:.1%} - Good")
                else:
                    st.warning(f"⚠️ **Win Rate**: {win_rate:.1%} - Review strategy")
            else:
                st.info("📊 Win rate analysis available after closing positions")
    
    else:
        st.info("📊 No trades match your current filters. Try adjusting the criteria above.")
    
    # Professional Tools
    st.markdown("---")
    st.markdown("### 🛠️ **Professional Analysis Tools**")
    
    tools_col1, tools_col2, tools_col3, tools_col4 = st.columns(4)
    
    with tools_col1:
        if st.button("📊 **Export Analysis**"):
            csv = filtered_trades.to_csv(index=False)
            st.download_button(
                label="💾 Download Professional CSV",
                data=csv,
                file_name=f"quantedge_professional_analysis_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv"
            )
    
    with tools_col2:
        if st.button("📈 **Performance Report**"):
            closed = filtered_trades[filtered_trades['pnl_dollars'].notna()]
            if not closed.empty:
                total_pnl = closed['pnl_dollars'].sum()
                win_rate = (closed['pnl_dollars'] > 0).mean()
                avg_win = closed[closed['pnl_dollars'] > 0]['pnl_dollars'].mean()
                avg_loss = closed[closed['pnl_dollars'] < 0]['pnl_dollars'].mean()
                
                st.success(f"""
                📊 **Professional Performance Report**
                • Total P&L: ${total_pnl:+.2f}
                • Win Rate: {win_rate:.1%}
                • Avg Win: ${avg_win:.2f}
                • Avg Loss: ${avg_loss:.2f}
                • Risk/Reward: {-avg_win/avg_loss:.2f}
                """)
            else:
                st.info("📊 Close some positions to generate performance report")
    
    with tools_col3:
        if st.button("🧠 **Strategy Analysis**"):
            strategy_performance = filtered_trades.groupby('strategy').agg({
                'confidence_level': 'mean',
                'pnl_dollars': ['count', 'sum', 'mean']
            }).round(2)
            
            st.dataframe(strategy_performance, use_container_width=True)
    
    with tools_col4:
        if st.button("🔄 **Refresh Data**"):
            st.rerun()

if __name__ == "__main__":
    render_professional_trade_journal()