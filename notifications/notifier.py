"""
Telegram notification service for trading signals.
Sends formatted signal messages to configured Telegram chat.
"""
import asyncio
import os
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv
from telegram import Bot
from telegram.constants import ParseMode

# Ensure .env is loaded
load_dotenv()


@dataclass
class SignalMessage:
    """Structured signal data for Telegram message."""
    date: str
    trade_decision: str
    fusion_state: str
    fusion_score: int
    fusion_confidence: str
    p_long: float
    p_short: float
    size_multiplier: float
    option_structures: str
    strike_guidance: str
    dte_range: str
    strategy_rationale: str
    tactical_put_active: bool = False
    tactical_put_strategy: str = ""


class TelegramNotifier:
    """
    Sends trading signal notifications via Telegram.
    
    Environment variables:
        TELEGRAM_BOT_TOKEN: Bot token from @BotFather
        TELEGRAM_CHAT_ID: Chat/user ID to send messages to
    
    Usage:
        notifier = TelegramNotifier()
        notifier.send_signal(signal_data)
    """
    
    def __init__(
        self,
        bot_token: Optional[str] = None,
        chat_id: Optional[str] = None,
    ):
        self.bot_token = bot_token or os.getenv("TELEGRAM_BOT_TOKEN")
        self.chat_id = chat_id or os.getenv("TELEGRAM_CHAT_ID")
        
        if not self.bot_token:
            raise ValueError("TELEGRAM_BOT_TOKEN not set")
        if not self.chat_id:
            raise ValueError("TELEGRAM_CHAT_ID not set")
        
        self.bot = Bot(token=self.bot_token)
    
    def _get_decision_emoji(self, decision: str) -> str:
        """Map trade decision to emoji."""
        return {
            "CALL": "🟢",
            "PUT": "🔴",
            "TACTICAL_PUT": "🟠",
            "NO_TRADE": "⚪",
        }.get(decision, "❓")
    
    def _get_confidence_emoji(self, confidence: str) -> str:
        """Map confidence level to emoji."""
        return {
            "HIGH": "🔥",
            "MEDIUM": "✨",
            "LOW": "💤",
        }.get(confidence, "")
    
    def _format_message(self, signal: SignalMessage) -> str:
        """Format signal data into Telegram message with markdown."""
        decision_emoji = self._get_decision_emoji(signal.trade_decision)
        confidence_emoji = self._get_confidence_emoji(signal.fusion_confidence)
        
        # Header
        msg = f"{decision_emoji} *{signal.trade_decision}* Signal\n"
        msg += f"📅 {signal.date}\n\n"
        
        # Market State
        msg += f"*Fusion State:* `{signal.fusion_state}`\n"
        msg += f"*Score:* {signal.fusion_score:+d} {confidence_emoji} ({signal.fusion_confidence})\n\n"
        
        # ML Probabilities
        msg += f"*ML Probabilities:*\n"
        msg += f"  📈 Long: `{signal.p_long:.1%}`\n"
        msg += f"  📉 Short: `{signal.p_short:.1%}`\n\n"
        
        # Position Sizing
        msg += f"*Size Multiplier:* `{signal.size_multiplier:.2f}x`\n\n"
        
        # Tactical Put (if active)
        if signal.tactical_put_active:
            msg += f"🛡️ *Tactical Put:* `{signal.tactical_put_strategy}`\n\n"
        
        # Option Strategy
        if signal.option_structures:
            msg += f"*Strategy:* `{signal.option_structures}`\n"
        if signal.strike_guidance:
            msg += f"*Strike:* `{signal.strike_guidance}`\n"
        if signal.dte_range:
            msg += f"*DTE:* `{signal.dte_range}`\n"
        
        # Rationale
        if signal.strategy_rationale:
            msg += f"\n💡 _{signal.strategy_rationale}_"
        
        return msg
    
    async def _send_async(self, message: str) -> bool:
        """Async send message via Telegram API."""
        try:
            await self.bot.send_message(
                chat_id=self.chat_id,
                text=message,
                parse_mode=ParseMode.MARKDOWN,
            )
            return True
        except Exception as e:
            print(f"Telegram send error: {e}")
            return False
    
    def send_signal(self, signal: SignalMessage) -> bool:
        """
        Send signal notification to Telegram.
        
        Returns:
            True if sent successfully, False otherwise.
        """
        if signal.trade_decision == "NO_TRADE":
            return False
        
        message = self._format_message(signal)
        return asyncio.run(self._send_async(message))
    
    def send_from_model(self, daily_signal) -> bool:
        """
        Send notification from DailySignal model instance.
        
        Args:
            daily_signal: DailySignal model instance
            
        Returns:
            True if sent, False if NO_TRADE or error.
        """
        signal = SignalMessage(
            date=str(daily_signal.date),
            trade_decision=daily_signal.trade_decision,
            fusion_state=daily_signal.fusion_state,
            fusion_score=daily_signal.fusion_score,
            fusion_confidence=daily_signal.fusion_confidence,
            p_long=daily_signal.p_long,
            p_short=daily_signal.p_short,
            size_multiplier=daily_signal.size_multiplier,
            option_structures=daily_signal.option_structures,
            strike_guidance=daily_signal.strike_guidance,
            dte_range=daily_signal.dte_range,
            strategy_rationale=daily_signal.strategy_rationale,
            tactical_put_active=daily_signal.tactical_put_active,
            tactical_put_strategy=daily_signal.tactical_put_strategy,
        )
        return self.send_signal(signal)
