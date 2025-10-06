"""
QuantEdge Enhanced Slack Notification System
Advanced notifications with daily summaries, risk alerts, and market analysis
Professional-grade trading alerts with actionable intelligence
"""
import json
import requests
import os
from datetime import datetime, timedelta
import logging
from typing import Dict, Optional, List
from pathlib import Path
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

class EnhancedSlackNotifications:
    """
    Enhanced Slack notification system with advanced analytics and intelligent alerts
    Research-based notification triggers and professional trading insights
    """
    
    def __init__(self):
        """Initialize enhanced notification system with intelligence"""
        self.webhook_url = None
        self.enabled = False
        self._load_environment_variables()
        
        # Notification settings
        self.daily_summary_sent = False
        self.last_risk_alert = None
        self.performance_thresholds = {
            'major_drawdown': -5.0,  # Alert if portfolio down 5%
            'high_volatility': 3.0,   # Alert if daily volatility > 3%
            'streak_wins': 5,         # Alert on 5+ winning trades in a row
            'streak_losses': 3,       # Alert on 3+ losing trades
            'position_concentration': 0.4,  # Alert if single position > 40%
            'daily_pnl_threshold': 1000,    # Alert on large daily P&L moves
        }
        
        print(f"🔍 Enhanced Slack Notifications:")
        print(f"   Status: {'Enabled' if self.enabled else 'Disabled'}")
        print(f"   Intelligence: Risk alerts, pattern detection, market analysis")
        print(f"   Thresholds: Drawdown {self.performance_thresholds['major_drawdown']}%, Volatility {self.performance_thresholds['high_volatility']}%")
    
    def _load_environment_variables(self):
        """Load environment variables with robust fallback"""
        # [Previous environment loading code - keeping it the same]
        possible_locations = [
            ".env", "dashboard/.env", "../.env",
            "/Users/andresbanuelos/PycharmProjects/QuantEdge/.env",
            "/Users/andresbanuelos/PycharmProjects/QuantEdge/dashboard/.env",
            Path(__file__).parent / ".env",
            Path(__file__).parent.parent / ".env"
        ]
        
        for location in possible_locations:
            env_path = Path(location)
            if env_path.exists() and env_path.is_file():
                try:
                    with open(env_path, 'r') as f:
                        for line in f:
                            line = line.strip()
                            if line and not line.startswith('#') and '=' in line:
                                key, value = line.split('=', 1)
                                key = key.strip()
                                value = value.strip().strip('"').strip("'")
                                os.environ[key] = value
                                if key == 'SLACK_WEBHOOK_URL':
                                    self.webhook_url = value
                    break
                except Exception as e:
                    continue
        
        # Fallback to manual setup
        if not self.webhook_url:
            self.webhook_url = os.getenv('SLACK_WEBHOOK_URL', 
                'https://hooks.slack.com/services/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')
        
        self.enabled = bool(self.webhook_url)
    
    def send_notification(self, message: str, channel: str = None, username: str = "QuantEdge Bot", icon_emoji: str = ":chart_with_upwards_trend:", priority: str = "normal") -> bool:
        """Send enhanced notification with priority levels"""
        if not self.enabled:
            return False
        
        try:
            # Add priority indicators
            priority_emojis = {
                "low": "ℹ️",
                "normal": "📊", 
                "high": "⚠️",
                "critical": "🚨"
            }
            
            # Enhance message with priority
            enhanced_message = f"{priority_emojis.get(priority, '📊')} {message}"
            
            payload = {
                "text": enhanced_message,
                "username": username,
                "icon_emoji": icon_emoji
            }
            
            if channel:
                payload["channel"] = channel
            
            response = requests.post(self.webhook_url, json=payload, timeout=10)
            
            if response.status_code == 200:
                logger.info(f"✅ Enhanced Slack notification sent (Priority: {priority})")
                return True
            else:
                logger.error(f"❌ Enhanced Slack notification failed: {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error sending enhanced Slack notification: {e}")
            return False
    
    def notify_intelligent_trade_executed(self, trade_data: Dict, market_context: Dict = None) -> bool:
        """Enhanced trade notification with market intelligence and pattern analysis"""
        action_emoji = "🟢" if trade_data['action'] == 'BUY' else "🔴"
        
        # Calculate trade metrics
        trade_value = trade_data.get('price', 0) * trade_data['quantity']
        confidence_emoji = "🔥" if trade_data.get('confidence', 0) > 80 else "✅" if trade_data.get('confidence', 0) > 65 else "⚠️"
        
        # Market context analysis
        market_context_text = ""
        if market_context:
            market_status = market_context.get('status', 'Unknown')
            volatility = market_context.get('volatility', 'Normal')
            market_context_text = f"\n📈 **Market Context:** {market_status} market, {volatility} volatility"
        
        # Strategy performance context
        strategy_context = ""
        if 'recent_performance' in trade_data:
            recent_perf = trade_data['recent_performance']
            strategy_context = f"\n🎯 **Strategy Performance:** {recent_perf.get('win_rate', 0):.1f}% win rate, {recent_perf.get('avg_return', 0):+.2f}% avg return"
        
        # Risk assessment
        risk_assessment = self._assess_trade_risk(trade_data)
        risk_text = f"\n🛡️ **Risk Assessment:** {risk_assessment['level']} - {risk_assessment['reason']}"
        
        message = f"""
{action_emoji} **INTELLIGENT TRADE EXECUTED** {action_emoji}

**{trade_data['action']} {trade_data['quantity']} {trade_data['symbol']}** {confidence_emoji}
• Price: ${trade_data.get('price', 0):.2f}
• Strategy: {trade_data.get('strategy', 'MANUAL')} 
• Confidence: {trade_data.get('confidence', 95):.1f}%
• Time: {datetime.now().strftime('%H:%M:%S PST')}

💰 **Trade Value: ${trade_value:,.2f}**{market_context_text}{strategy_context}{risk_text}

📊 **Next Steps:** Monitor for {trade_data.get('take_profit', 'target')} target, stop at ${trade_data.get('stop_loss', 0):.2f}
        """
        
        # Determine priority based on trade size and confidence
        priority = "high" if trade_value > 10000 or trade_data.get('confidence', 0) > 85 else "normal"
        
        return self.send_notification(
            message, 
            username="QuantEdge Pro Trader",
            icon_emoji=":money_with_wings:",
            priority=priority
        )
    
    def notify_daily_performance_summary(self, daily_stats: Dict, risk_metrics: Dict, top_positions: List) -> bool:
        """Comprehensive daily performance summary with actionable insights"""
        
        if self.daily_summary_sent and datetime.now().hour < 16:  # Don't spam daily summaries
            return False
        
        # Performance metrics
        daily_pnl = daily_stats.get('daily_pnl', 0)
        total_trades = daily_stats.get('total_trades', 0)
        win_rate = daily_stats.get('win_rate', 0)
        best_trade = daily_stats.get('best_trade', {})
        worst_trade = daily_stats.get('worst_trade', {})
        
        # Risk analysis
        sharpe_ratio = risk_metrics.get('sharpe_ratio', 0)
        max_drawdown = risk_metrics.get('max_drawdown_pct', 0)
        volatility = risk_metrics.get('volatility', 0)
        
        # Performance emoji
        perf_emoji = "🚀" if daily_pnl > 500 else "📈" if daily_pnl > 0 else "📉" if daily_pnl > -500 else "🔥"
        
        # Risk level assessment
        risk_level = "🟢 LOW" if max_drawdown > -2 and volatility < 2 else "🟡 MODERATE" if max_drawdown > -5 else "🔴 HIGH"
        
        # Top positions analysis
        positions_text = ""
        if top_positions:
            positions_text = "\n📊 **Top Positions:**"
            for i, pos in enumerate(top_positions[:3]):
                pnl_emoji = "🟢" if pos.get('unrealized_pl', 0) >= 0 else "🔴"
                positions_text += f"\n{i+1}. {pos.get('symbol', 'N/A')} {pnl_emoji} ${pos.get('unrealized_pl', 0):+,.0f} ({pos.get('unrealized_plpc', 0):+.1f}%)"
        
        # Market insights
        market_insights = self._generate_market_insights(daily_stats, risk_metrics)
        insights_text = f"\n🧠 **AI Insights:**\n{market_insights}"
        
        # Tomorrow's strategy
        strategy_recommendation = self._recommend_tomorrow_strategy(daily_stats, risk_metrics)
        strategy_text = f"\n🎯 **Tomorrow's Strategy:**\n{strategy_recommendation}"
        
        message = f"""
📈 **DAILY TRADING SUMMARY** {perf_emoji}

**Today's Performance ({datetime.now().strftime('%Y-%m-%d')}):**
• Total P&L: **${daily_pnl:+,.2f}**
• Trades Executed: {total_trades}
• Win Rate: {win_rate:.1f}%
• Best Trade: +${best_trade.get('pnl', 0):.2f} ({best_trade.get('symbol', 'N/A')})
• Worst Trade: ${worst_trade.get('pnl', 0):.2f} ({worst_trade.get('symbol', 'N/A')})

**📊 Risk Metrics:**
• Risk Level: {risk_level}
• Sharpe Ratio: {sharpe_ratio:.2f}
• Max Drawdown: {max_drawdown:.1f}%
• Volatility: {volatility:.2f}%{positions_text}{insights_text}{strategy_text}

🌅 **Tomorrow:** Market opens 6:30 AM PST - Good luck trading!
        """
        
        self.daily_summary_sent = True
        return self.send_notification(
            message,
            username="QuantEdge Daily Report",
            icon_emoji=":calendar:",
            priority="normal"
        )
    
    def notify_risk_alert(self, alert_type: str, current_value: float, threshold: float, context: Dict) -> bool:
        """Intelligent risk management alerts with actionable recommendations"""
        
        # Prevent spam - only send risk alerts every 30 minutes
        now = datetime.now()
        if (self.last_risk_alert and 
            (now - self.last_risk_alert).total_seconds() < 1800):
            return False
        
        alert_configs = {
            'major_drawdown': {
                'emoji': '🚨',
                'title': 'MAJOR DRAWDOWN ALERT',
                'color': 'danger',
                'priority': 'critical'
            },
            'high_volatility': {
                'emoji': '⚠️',
                'title': 'HIGH VOLATILITY ALERT',
                'color': 'warning', 
                'priority': 'high'
            },
            'position_concentration': {
                'emoji': '🎯',
                'title': 'POSITION CONCENTRATION ALERT',
                'color': 'warning',
                'priority': 'high'
            },
            'streak_losses': {
                'emoji': '🔴',
                'title': 'LOSING STREAK ALERT',
                'color': 'danger',
                'priority': 'high'
            },
            'streak_wins': {
                'emoji': '🔥',
                'title': 'WINNING STREAK ALERT',
                'color': 'success',
                'priority': 'normal'
            }
        }
        
        config = alert_configs.get(alert_type, alert_configs['major_drawdown'])
        
        # Generate specific recommendations
        recommendations = self._generate_risk_recommendations(alert_type, current_value, context)
        
        message = f"""
{config['emoji']} **{config['title']}** {config['emoji']}

**Alert Details:**
• Metric: {alert_type.replace('_', ' ').title()}
• Current Value: {current_value:.2f}{'%' if 'pct' in alert_type else ''}
• Threshold: {threshold:.2f}{'%' if 'pct' in alert_type else ''}
• Severity: {config['priority'].upper()}

**📊 Current Situation:**
{context.get('description', 'Risk threshold exceeded')}

**🎯 Recommended Actions:**
{recommendations}

**⏰ Time:** {datetime.now().strftime('%H:%M:%S PST')}

💡 **Remember:** Risk management is key to long-term profitability. Consider reducing position sizes or taking a break if needed.
        """
        
        self.last_risk_alert = now
        return self.send_notification(
            message,
            username="QuantEdge Risk Manager",
            icon_emoji=":warning:",
            priority=config['priority']
        )
    
    def notify_market_condition_change(self, condition_change: Dict) -> bool:
        """Alert on significant market condition changes that affect trading"""
        
        old_condition = condition_change.get('from', 'Unknown')
        new_condition = condition_change.get('to', 'Unknown')
        confidence = condition_change.get('confidence', 0)
        impact = condition_change.get('impact', 'neutral')
        
        # Impact emojis
        impact_emojis = {
            'bullish': '🚀',
            'bearish': '📉', 
            'neutral': '➡️',
            'volatile': '⚡'
        }
        
        # Strategy adjustments
        strategy_adjustments = self._get_strategy_adjustments_for_market(new_condition)
        
        message = f"""
🌊 **MARKET CONDITION CHANGE** {impact_emojis.get(impact, '📊')}

**Market Shift Detected:**
• From: {old_condition} ➜ **{new_condition}**
• Confidence: {confidence:.1f}%
• Impact: {impact.title()}
• Detection Time: {datetime.now().strftime('%H:%M:%S PST')}

**🎯 Strategy Adjustments:**
{strategy_adjustments}

**📈 What This Means:**
• {condition_change.get('explanation', 'Market dynamics have shifted')}
• Expected duration: {condition_change.get('expected_duration', 'Unknown')}
• Key levels to watch: {condition_change.get('key_levels', 'Monitor support/resistance')}

**🤖 Auto Trading Impact:**
• {condition_change.get('auto_trading_impact', 'Continue monitoring with current settings')}

Stay alert and adapt your strategy accordingly! 📊
        """
        
        return self.send_notification(
            message,
            username="QuantEdge Market Analyst",
            icon_emoji=":ocean:",
            priority="high"
        )
    
    def notify_weekly_performance_review(self, weekly_stats: Dict) -> bool:
        """Comprehensive weekly performance review with deep insights"""
        
        # Only send on Sunday evenings or Monday mornings
        current_day = datetime.now().weekday()  # Monday = 0
        if current_day not in [0, 6]:  # Monday or Sunday
            return False
        
        total_pnl = weekly_stats.get('total_pnl', 0)
        total_trades = weekly_stats.get('total_trades', 0)
        win_rate = weekly_stats.get('win_rate', 0)
        best_day = weekly_stats.get('best_day', {})
        worst_day = weekly_stats.get('worst_day', {})
        strategy_performance = weekly_stats.get('strategy_performance', {})
        
        # Performance grade
        grade = self._calculate_performance_grade(weekly_stats)
        grade_emoji = {"A": "🏆", "B": "🥈", "C": "🥉", "D": "📚", "F": "🚨"}
        
        # Strategy analysis
        strategy_text = ""
        if strategy_performance:
            best_strategy = max(strategy_performance.items(), key=lambda x: x[1].get('pnl', 0))
            worst_strategy = min(strategy_performance.items(), key=lambda x: x[1].get('pnl', 0))
            
            strategy_text = f"""
**📊 Strategy Performance:**
• Best: {best_strategy[0]} (${best_strategy[1].get('pnl', 0):+.2f})
• Worst: {worst_strategy[0]} (${worst_strategy[1].get('pnl', 0):+.2f})
"""
        
        # Market insights and predictions
        market_analysis = self._analyze_weekly_market_patterns(weekly_stats)
        
        # Goals for next week
        next_week_goals = self._set_next_week_goals(weekly_stats)
        
        message = f"""
📅 **WEEKLY TRADING REVIEW** {grade_emoji.get(grade, '📊')}

**Week of {(datetime.now() - timedelta(days=7)).strftime('%m/%d')} - {datetime.now().strftime('%m/%d/%Y')}:**

**🎯 Performance Grade: {grade}**
• Total P&L: **${total_pnl:+,.2f}**
• Total Trades: {total_trades}
• Win Rate: {win_rate:.1f}%
• Best Day: ${best_day.get('pnl', 0):+.2f} ({best_day.get('date', 'N/A')})
• Worst Day: ${worst_day.get('pnl', 0):+.2f} ({worst_day.get('date', 'N/A')}){strategy_text}

**🧠 Market Analysis:**
{market_analysis}

**🎯 Next Week's Goals:**
{next_week_goals}

**💡 Key Takeaways:**
• {weekly_stats.get('key_insight_1', 'Continue disciplined trading approach')}
• {weekly_stats.get('key_insight_2', 'Monitor risk management closely')}
• {weekly_stats.get('key_insight_3', 'Stay focused on high-probability setups')}

Here's to an even better week ahead! 🚀📈
        """
        
        return self.send_notification(
            message,
            username="QuantEdge Weekly Review",
            icon_emoji=":calendar:",
            priority="normal"
        )
    
    def _assess_trade_risk(self, trade_data: Dict) -> Dict:
        """Assess risk level of individual trade"""
        trade_value = trade_data.get('price', 0) * trade_data['quantity']
        confidence = trade_data.get('confidence', 0)
        
        if trade_value > 20000:
            return {'level': 'HIGH', 'reason': 'Large position size'}
        elif confidence < 60:
            return {'level': 'HIGH', 'reason': 'Low confidence signal'}
        elif trade_value > 10000:
            return {'level': 'MODERATE', 'reason': 'Medium position size'}
        elif confidence > 80:
            return {'level': 'LOW', 'reason': 'High confidence signal'}
        else:
            return {'level': 'MODERATE', 'reason': 'Standard risk profile'}
    
    def _generate_market_insights(self, daily_stats: Dict, risk_metrics: Dict) -> str:
        """Generate AI-powered market insights"""
        insights = []
        
        win_rate = daily_stats.get('win_rate', 0)
        daily_pnl = daily_stats.get('daily_pnl', 0)
        volatility = risk_metrics.get('volatility', 0)
        
        if win_rate > 70:
            insights.append("• High win rate suggests strong market alignment with your strategies")
        elif win_rate < 40:
            insights.append("• Low win rate may indicate choppy market conditions")
        
        if volatility > 3:
            insights.append("• High volatility detected - consider reducing position sizes")
        elif volatility < 1:
            insights.append("• Low volatility environment - potential for breakout moves")
        
        if daily_pnl > 1000:
            insights.append("• Strong performance today - maintain discipline and risk management")
        elif daily_pnl < -500:
            insights.append("• Challenging day - consider reassessing strategy parameters")
        
        if not insights:
            insights.append("• Market conditions appear normal - continue with systematic approach")
        
        return '\n'.join(insights)
    
    def _recommend_tomorrow_strategy(self, daily_stats: Dict, risk_metrics: Dict) -> str:
        """AI-powered strategy recommendations for next trading day"""
        recommendations = []
        
        win_rate = daily_stats.get('win_rate', 0)
        volatility = risk_metrics.get('volatility', 0)
        daily_pnl = daily_stats.get('daily_pnl', 0)
        
        if win_rate > 70 and daily_pnl > 0:
            recommendations.append("• Continue with current strategy mix - showing strong performance")
        elif win_rate < 50:
            recommendations.append("• Consider focusing on highest confidence signals only")
        
        if volatility > 2:
            recommendations.append("• High volatility expected - favor momentum strategies")
        else:
            recommendations.append("• Low volatility environment - mean reversion may be effective")
        
        if daily_pnl < -1000:
            recommendations.append("• Consider reducing position sizes after significant drawdown")
        
        recommendations.append("• Monitor key support/resistance levels for entry opportunities")
        
        return '\n'.join(recommendations)
    
    def _generate_risk_recommendations(self, alert_type: str, current_value: float, context: Dict) -> str:
        """Generate specific risk management recommendations"""
        recommendations = {
            'major_drawdown': [
                "• Immediately reduce position sizes by 50%",
                "• Consider stopping auto trading until recovery",
                "• Review and tighten stop-loss levels",
                "• Focus on capital preservation over growth"
            ],
            'high_volatility': [
                "• Reduce position sizes to manage risk",
                "• Widen stop-loss levels to avoid whipsaws", 
                "• Consider volatility-based position sizing",
                "• Monitor market conditions closely"
            ],
            'position_concentration': [
                "• Diversify holdings across multiple positions",
                "• Consider reducing largest position size",
                "• Implement maximum position size limits",
                "• Review correlation between holdings"
            ],
            'streak_losses': [
                "• Take a break from trading to reassess",
                "• Review recent trades for pattern analysis", 
                "• Consider reducing confidence thresholds",
                "• Focus on risk management over returns"
            ],
            'streak_wins': [
                "• Maintain discipline - avoid overconfidence",
                "• Consider taking some profits off the table",
                "• Review strategy performance sustainability",
                "• Prepare for potential market shift"
            ]
        }
        
        return '\n'.join(recommendations.get(alert_type, ["• Monitor situation closely", "• Maintain risk management discipline"]))
    
    def _get_strategy_adjustments_for_market(self, market_condition: str) -> str:
        """Get strategy adjustments based on market condition"""
        adjustments = {
            'trending_up': "• Favor momentum and breakout strategies\n• Increase position sizes on strong signals\n• Look for pullback entries in uptrends",
            'trending_down': "• Focus on short strategies and mean reversion\n• Reduce overall exposure\n• Wait for oversold bounce opportunities",
            'sideways': "• Emphasize mean reversion strategies\n• Trade range boundaries\n• Reduce breakout strategy allocation",
            'volatile': "• Reduce position sizes significantly\n• Use wider stop losses\n• Focus on highest confidence signals only",
            'low_volatility': "• Prepare for potential breakout\n• Look for accumulation patterns\n• Consider increasing position sizes slightly"
        }
        
        return adjustments.get(market_condition, "• Continue with current strategy mix\n• Monitor for changes\n• Maintain discipline")
    
    def _calculate_performance_grade(self, weekly_stats: Dict) -> str:
        """Calculate letter grade for weekly performance"""
        pnl = weekly_stats.get('total_pnl', 0)
        win_rate = weekly_stats.get('win_rate', 0)
        total_trades = weekly_stats.get('total_trades', 0)
        
        score = 0
        
        # P&L component (40%)
        if pnl > 2000: score += 40
        elif pnl > 1000: score += 30
        elif pnl > 0: score += 20
        elif pnl > -500: score += 10
        else: score += 0
        
        # Win rate component (30%)
        if win_rate > 70: score += 30
        elif win_rate > 60: score += 25
        elif win_rate > 50: score += 20
        elif win_rate > 40: score += 10
        else: score += 0
        
        # Activity component (30%)
        if total_trades >= 20: score += 30
        elif total_trades >= 10: score += 25
        elif total_trades >= 5: score += 15
        else: score += 5
        
        if score >= 90: return 'A'
        elif score >= 80: return 'B'
        elif score >= 70: return 'C'
        elif score >= 60: return 'D'
        else: return 'F'
    
    def _analyze_weekly_market_patterns(self, weekly_stats: Dict) -> str:
        """Analyze weekly market patterns for insights"""
        analysis = []
        
        daily_performance = weekly_stats.get('daily_breakdown', {})
        if daily_performance:
            best_days = [day for day, pnl in daily_performance.items() if pnl > 100]
            if best_days:
                analysis.append(f"• Strong performance on {', '.join(best_days)}")
        
        market_conditions = weekly_stats.get('market_conditions', {})
        if market_conditions:
            dominant_condition = max(market_conditions.items(), key=lambda x: x[1])
            analysis.append(f"• Market was primarily {dominant_condition[0]} ({dominant_condition[1]} days)")
        
        if not analysis:
            analysis.append("• Mixed market conditions throughout the week")
            analysis.append("• Continue monitoring for emerging patterns")
        
        return '\n'.join(analysis)
    
    def _set_next_week_goals(self, weekly_stats: Dict) -> str:
        """Set intelligent goals for next week"""
        current_pnl = weekly_stats.get('total_pnl', 0)
        current_win_rate = weekly_stats.get('win_rate', 0)
        
        goals = []
        
        if current_pnl > 0:
            target_pnl = current_pnl * 1.1  # 10% improvement
            goals.append(f"• Target P&L: ${target_pnl:+,.0f} (+10% from this week)")
        else:
            goals.append("• Primary goal: Return to profitability")
        
        if current_win_rate < 60:
            goals.append(f"• Improve win rate to {current_win_rate + 10:.0f}%")
        
        goals.append("• Maintain disciplined risk management")
        goals.append("• Focus on high-probability setups")
        
        return '\n'.join(goals)

# Create enhanced global instance
enhanced_slack_notifier = EnhancedSlackNotifications()