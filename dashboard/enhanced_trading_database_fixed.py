"""
Enhanced Trading Database - Final Clean Version
Handles all database operations, real vs sample data tracking, and performance metrics
NO database column errors, full migration support
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sqlite3
from typing import Dict, List, Tuple
from dataclasses import dataclass
import os

@dataclass
class BlendedPerformanceMetrics:
    """Performance metrics that blend sample and real data"""
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    avg_win_loss: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    total_pnl: float
    daily_pnl: float
    real_trades_count: int
    sample_trades_count: int
    real_data_percentage: float
    real_pnl: float
    sample_pnl: float

class EnhancedTradingDatabase:
    """Clean, production-ready trading database with real data accumulation"""
    
    def __init__(self, db_path: str = "quantedge_trading.db"):
        self.db_path = db_path
        self.init_enhanced_database()
    
    def init_enhanced_database(self):
        """Initialize database with migration support for existing databases"""
        if os.path.exists(self.db_path):
            self.migrate_existing_database()
        else:
            self.create_fresh_database()
        
        with sqlite3.connect(self.db_path) as conn:
            trade_count = conn.execute("SELECT COUNT(*) FROM trades").fetchone()[0]
            if trade_count == 0:
                self._create_baseline_sample_data(conn)
                self._update_data_composition(conn)
    
    def migrate_existing_database(self):
        """Migrate existing database to new schema without errors"""
        print("🔄 Migrating existing database...")
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("PRAGMA table_info(trades)")
            columns = [column[1] for column in cursor.fetchall()]
            
            if 'data_source' not in columns:
                conn.execute("ALTER TABLE trades ADD COLUMN data_source TEXT DEFAULT 'sample'")
            if 'trade_source' not in columns:
                conn.execute("ALTER TABLE trades ADD COLUMN trade_source TEXT DEFAULT 'baseline'")
            
            cursor = conn.execute("PRAGMA table_info(portfolio_history)")
            columns = [column[1] for column in cursor.fetchall()]
            if 'data_source' not in columns:
                conn.execute("ALTER TABLE portfolio_history ADD COLUMN data_source TEXT DEFAULT 'sample'")
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS data_composition (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date DATE DEFAULT CURRENT_DATE,
                    real_trades_count INTEGER DEFAULT 0,
                    sample_trades_count INTEGER DEFAULT 0,
                    real_data_percentage REAL DEFAULT 0,
                    total_real_pnl REAL DEFAULT 0,
                    total_sample_pnl REAL DEFAULT 0,
                    last_updated DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS account_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date DATE DEFAULT CURRENT_DATE,
                    account_equity REAL,
                    buying_power REAL,
                    cash REAL,
                    portfolio_value REAL,
                    day_trade_count INTEGER DEFAULT 0,
                    account_status TEXT,
                    data_source TEXT DEFAULT 'alpaca'
                )
            """)
            
            conn.commit()
            print("✅ Database migration completed successfully!")
    
    def create_fresh_database(self):
        """Create completely fresh database"""
        print("🏗️ Creating fresh database...")
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    action TEXT NOT NULL,
                    quantity INTEGER NOT NULL,
                    price REAL NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    strategy TEXT,
                    confidence REAL,
                    pnl REAL DEFAULT 0,
                    commission REAL DEFAULT 1.00,
                    status TEXT DEFAULT 'executed',
                    data_source TEXT DEFAULT 'real',
                    trade_source TEXT DEFAULT 'manual'
                )
            """)
            
            conn.execute("""
                CREATE TABLE portfolio_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date DATE NOT NULL,
                    total_value REAL NOT NULL,
                    cash REAL NOT NULL,
                    positions_value REAL NOT NULL,
                    daily_pnl REAL NOT NULL,
                    daily_pnl_pct REAL NOT NULL,
                    drawdown REAL DEFAULT 0,
                    num_positions INTEGER DEFAULT 0,
                    data_source TEXT DEFAULT 'real',
                    UNIQUE(date)
                )
            """)
            
            conn.execute("""
                CREATE TABLE data_composition (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date DATE DEFAULT CURRENT_DATE,
                    real_trades_count INTEGER DEFAULT 0,
                    sample_trades_count INTEGER DEFAULT 0,
                    real_data_percentage REAL DEFAULT 0,
                    total_real_pnl REAL DEFAULT 0,
                    total_sample_pnl REAL DEFAULT 0,
                    last_updated DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.execute("""
                CREATE TABLE account_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date DATE DEFAULT CURRENT_DATE,
                    account_equity REAL,
                    buying_power REAL,
                    cash REAL,
                    portfolio_value REAL,
                    day_trade_count INTEGER DEFAULT 0,
                    account_status TEXT,
                    data_source TEXT DEFAULT 'alpaca'
                )
            """)
            
            conn.commit()
            print("✅ Fresh database created!")
    
    def _create_baseline_sample_data(self, conn):
        """Create realistic baseline sample data"""
        print("📊 Creating baseline sample data...")
        
        symbols = ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'TSLA', 'AMD', 'META', 'AMZN', 'JPM', 'BAC']
        strategies = ['MOMENTUM', 'BREAKOUT', 'MEAN_REVERSION', 'TREND_FOLLOWING']
        
        start_date = datetime.now() - timedelta(days=90)
        trades_data = []
        portfolio_data = []
        
        np.random.seed(42)
        current_portfolio_value = 100000
        
        for day in range(90):
            current_date = start_date + timedelta(days=day)
            
            if current_date.weekday() >= 5:
                continue
            
            num_trades = np.random.poisson(1.2)
            daily_pnl = 0
            
            for _ in range(num_trades):
                symbol = np.random.choice(symbols)
                action = np.random.choice(['BUY', 'SELL'])
                quantity = np.random.randint(10, 200)
                price = np.random.uniform(50, 500)
                strategy = np.random.choice(strategies)
                confidence = np.random.uniform(65, 95)
                
                if confidence > 80:
                    pnl = np.random.uniform(50, 300)
                elif confidence > 70:
                    pnl = np.random.uniform(-50, 200)
                else:
                    pnl = np.random.uniform(-150, 100)
                
                commission = quantity * 0.005
                pnl -= commission
                
                trades_data.append((
                    symbol, action, quantity, price, 
                    current_date.strftime('%Y-%m-%d %H:%M:%S'),
                    strategy, confidence, pnl, commission, 'executed',
                    'sample', 'baseline'
                ))
                
                daily_pnl += pnl
            
            current_portfolio_value += daily_pnl
            peak_value = max(current_portfolio_value, 100000)
            drawdown = (peak_value - current_portfolio_value) / peak_value * 100
            
            portfolio_data.append((
                current_date.strftime('%Y-%m-%d'),
                current_portfolio_value,
                current_portfolio_value * 0.1,
                current_portfolio_value * 0.9,
                daily_pnl,
                (daily_pnl / current_portfolio_value) * 100,
                drawdown,
                np.random.randint(3, 12),
                'sample'
            ))
        
        conn.executemany("""
            INSERT INTO trades 
            (symbol, action, quantity, price, timestamp, strategy, confidence, pnl, commission, status, data_source, trade_source)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, trades_data)
        
        conn.executemany("""
            INSERT OR REPLACE INTO portfolio_history 
            (date, total_value, cash, positions_value, daily_pnl, daily_pnl_pct, drawdown, num_positions, data_source)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, portfolio_data)
        
        conn.commit()
        print(f"✅ Created {len(trades_data)} baseline sample trades")
    
    def log_real_trade(self, trade_data: Dict):
        """Log a REAL trade from actual trading"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO trades 
                (symbol, action, quantity, price, strategy, confidence, pnl, commission, data_source, trade_source)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'real', 'live_trading')
            """, (
                trade_data['symbol'], trade_data['action'], trade_data['quantity'],
                trade_data['price'], trade_data.get('strategy'), trade_data.get('confidence'),
                trade_data.get('pnl', 0), trade_data.get('commission', 0)
            ))
            
            self._update_data_composition(conn)
            print(f"📈 REAL TRADE LOGGED: {trade_data['action']} {trade_data['quantity']} {trade_data['symbol']}")
    
    def log_alpaca_account_snapshot(self, account_data: Dict):
        """Log Alpaca account data"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO account_history 
                (date, account_equity, buying_power, cash, portfolio_value, account_status, data_source)
                VALUES (CURRENT_DATE, ?, ?, ?, ?, ?, 'alpaca')
            """, (
                account_data.get('equity', 0),
                account_data.get('buying_power', 0), 
                account_data.get('cash', 0),
                account_data.get('portfolio_value', 0),
                account_data.get('status', 'ACTIVE')
            ))
            conn.commit()
    
    def simulate_real_trade_execution(self, symbol: str, action: str, quantity: int, price: float, strategy: str = "LIVE"):
        """Simulate executing a real trade for demo purposes"""
        if action.upper() == 'BUY':
            price_change = np.random.uniform(-0.02, 0.03)
            pnl = quantity * price * price_change
        else:
            pnl = np.random.uniform(50, 500)
        
        commission = quantity * 0.005
        net_pnl = pnl - commission
        
        trade_data = {
            'symbol': symbol,
            'action': action,
            'quantity': quantity,
            'price': price,
            'strategy': strategy,
            'confidence': 85.0,
            'pnl': net_pnl,
            'commission': commission
        }
        
        self.log_real_trade(trade_data)
        return trade_data
    
    def _update_data_composition(self, conn):
        """Update data composition tracking"""
        real_trades = conn.execute("SELECT COUNT(*) FROM trades WHERE data_source = 'real'").fetchone()[0]
        sample_trades = conn.execute("SELECT COUNT(*) FROM trades WHERE data_source = 'sample'").fetchone()[0]
        
        total_trades = real_trades + sample_trades
        real_percentage = (real_trades / total_trades * 100) if total_trades > 0 else 0
        
        real_pnl = conn.execute("SELECT COALESCE(SUM(pnl), 0) FROM trades WHERE data_source = 'real'").fetchone()[0]
        sample_pnl = conn.execute("SELECT COALESCE(SUM(pnl), 0) FROM trades WHERE data_source = 'sample'").fetchone()[0]
        
        conn.execute("""
            INSERT OR REPLACE INTO data_composition 
            (date, real_trades_count, sample_trades_count, real_data_percentage, total_real_pnl, total_sample_pnl, last_updated)
            VALUES (CURRENT_DATE, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
        """, (real_trades, sample_trades, real_percentage, real_pnl, sample_pnl))
        
        conn.commit()
    
    def calculate_blended_performance_metrics(self) -> BlendedPerformanceMetrics:
        """Calculate performance metrics with real vs sample data composition"""
        with sqlite3.connect(self.db_path) as conn:
            trades_df = pd.read_sql_query("""
                SELECT *, CASE WHEN data_source = 'real' THEN 1 ELSE 0 END as is_real_trade
                FROM trades ORDER BY timestamp
            """, conn)
            
            portfolio_df = pd.read_sql_query("""
                SELECT * FROM portfolio_history ORDER BY date DESC LIMIT 90
            """, conn)
            
            if trades_df.empty:
                return BlendedPerformanceMetrics(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
            
            real_trades = trades_df[trades_df['data_source'] == 'real']
            sample_trades = trades_df[trades_df['data_source'] == 'sample']
            
            total_trades = len(trades_df)
            winning_trades = len(trades_df[trades_df['pnl'] > 0])
            losing_trades = len(trades_df[trades_df['pnl'] < 0])
            
            win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
            total_pnl = trades_df['pnl'].sum()
            
            real_pnl = real_trades['pnl'].sum() if not real_trades.empty else 0
            sample_pnl = sample_trades['pnl'].sum() if not sample_trades.empty else 0
            
            avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
            avg_loss = trades_df[trades_df['pnl'] < 0]['pnl'].mean() if losing_trades > 0 else 0
            avg_win_loss = abs(avg_win / avg_loss) if avg_loss != 0 else 0
            
            total_wins = trades_df[trades_df['pnl'] > 0]['pnl'].sum()
            total_losses = abs(trades_df[trades_df['pnl'] < 0]['pnl'].sum())
            profit_factor = (total_wins / total_losses) if total_losses > 0 else float('inf')
            
            if not portfolio_df.empty:
                portfolio_df = portfolio_df.sort_values('date')
                initial_value = portfolio_df.iloc[0]['total_value']
                final_value = portfolio_df.iloc[-1]['total_value']
                total_return = ((final_value - initial_value) / initial_value * 100) if initial_value > 0 else 0
                
                days_traded = len(portfolio_df)
                annualized_return = (((final_value / initial_value) ** (252 / days_traded)) - 1) * 100 if days_traded > 0 else 0
                
                max_drawdown = portfolio_df['drawdown'].max()
                
                daily_returns = portfolio_df['daily_pnl_pct']
                sharpe_ratio = (daily_returns.mean() / daily_returns.std() * np.sqrt(252)) if daily_returns.std() > 0 else 0
                
                today_pnl = portfolio_df.iloc[-1]['daily_pnl'] if not portfolio_df.empty else 0
            else:
                total_return = annualized_return = max_drawdown = sharpe_ratio = today_pnl = 0
            
            real_trades_count = len(real_trades)
            sample_trades_count = len(sample_trades)
            real_data_percentage = (real_trades_count / total_trades * 100) if total_trades > 0 else 0
            
            return BlendedPerformanceMetrics(
                total_return=total_return,
                annualized_return=annualized_return,
                sharpe_ratio=sharpe_ratio,
                max_drawdown=max_drawdown,
                win_rate=win_rate,
                profit_factor=profit_factor,
                avg_win_loss=avg_win_loss,
                total_trades=total_trades,
                winning_trades=winning_trades,
                losing_trades=losing_trades,
                total_pnl=total_pnl,
                daily_pnl=today_pnl,
                real_trades_count=real_trades_count,
                sample_trades_count=sample_trades_count,
                real_data_percentage=real_data_percentage,
                real_pnl=real_pnl,
                sample_pnl=sample_pnl
            )
    
    def get_data_composition_summary(self) -> Dict:
        """Get summary of data composition for dashboard"""
        metrics = self.calculate_blended_performance_metrics()
        
        return {
            'real_trades': metrics.real_trades_count,
            'sample_trades': metrics.sample_trades_count,
            'real_percentage': metrics.real_data_percentage,
            'real_pnl': metrics.real_pnl,
            'sample_pnl': metrics.sample_pnl,
            'status': self._get_data_status(metrics.real_data_percentage),
            'next_milestone': self._get_next_milestone(metrics.real_trades_count)
        }
    
    def _get_data_status(self, real_percentage: float) -> str:
        """Get current data status description"""
        if real_percentage == 0:
            return "🎯 Fresh Start - Ready for real paper trading"
        elif real_percentage < 10:
            return "🌱 Early Stage - Your real trades are accumulating"
        elif real_percentage < 25:
            return "📈 Growing Real Data - Sample data still dominates"
        elif real_percentage < 50:
            return "⚡ Significant Real Data - Your performance emerging"
        elif real_percentage < 75:
            return "🚀 Real Data Dominant - Mostly your actual performance"
        else:
            return "💎 Pure Performance - Almost entirely your real data"
    
    def _get_next_milestone(self, real_trades: int) -> str:
        """Get next milestone message"""
        if real_trades < 10:
            return f"Next: {10 - real_trades} more real trades for first milestone"
        elif real_trades < 50:
            return f"Next: {50 - real_trades} more real trades for statistical significance"
        elif real_trades < 100:
            return f"Next: {100 - real_trades} more real trades for robust analysis"
        elif real_trades < 250:
            return f"Next: {250 - real_trades} more real trades for extensive history"
        else:
            return "🏆 Milestone achieved: Extensive real trading history!"
    
    def get_real_vs_sample_breakdown(self) -> pd.DataFrame:
        """Get breakdown of performance by data source"""
        with sqlite3.connect(self.db_path) as conn:
            return pd.read_sql_query("""
                SELECT 
                    data_source,
                    COUNT(*) as trade_count,
                    SUM(pnl) as total_pnl,
                    AVG(pnl) as avg_pnl,
                    SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as winning_trades,
                    SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) * 100.0 / COUNT(*) as win_rate,
                    MAX(pnl) as best_trade,
                    MIN(pnl) as worst_trade
                FROM trades 
                GROUP BY data_source
                ORDER BY data_source
            """, conn)
    
    def add_sample_real_trade_for_demo(self, symbol: str = "AAPL"):
        """Add a sample 'real' trade for demonstration"""
        return self.simulate_real_trade_execution(
            symbol=symbol,
            action="BUY", 
            quantity=100,
            price=175.50,
            strategy="DEMO"
        )