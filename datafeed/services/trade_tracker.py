"""
Option Trade Tracker

Manages OptionTrade lifecycle: entry, monitoring, exit, and PnL attribution.
Computes MAE/MFE from snapshots during hold period.

Usage:
    from datafeed.services.trade_tracker import TradeTracker
    
    tracker = TradeTracker()
    
    # Open a trade (live)
    trade = tracker.open_trade(
        signal_type='BULL_PROBE',
        direction='LONG',
        symbol='BTC-17APR26-68000-C',
        qty=Decimal('0.1'),
        entry_price=Decimal('1700'),
        entry_spot=Decimal('67000'),
    )
    
    # Open a trade (historical backfill)
    trade = tracker.open_trade(
        signal_type='BULL_PROBE',
        direction='LONG',
        symbol='BTC-17APR26-68000-C',
        qty=Decimal('0.1'),
        entry_price=Decimal('1700'),
        entry_spot=Decimal('67000'),
        entry_timestamp=datetime(2026, 4, 1, 10, 0, tzinfo=timezone.utc),
    )
    
    # Close a trade
    tracker.close_trade(
        trade_id=trade.trade_id,
        exit_price=Decimal('2200'),
        exit_spot=Decimal('69000'),
        exit_reason='tp',
    )
"""
import uuid
from datetime import datetime, timezone
from decimal import Decimal
from typing import Optional

from django.db.models import Max, Min

from datafeed.models import OptionSnapshot, OptionTrade


class TradeTracker:
    """
    Manages option trade lifecycle and computes path statistics.
    """
    
    def open_trade(
        self,
        signal_type: str,
        direction: str,
        symbol: str,
        qty: Decimal,
        entry_price: Decimal,
        entry_spot: Decimal,
        entry_iv: Optional[Decimal] = None,
        notional: Optional[Decimal] = None,
        entry_timestamp: Optional[datetime] = None,
        is_paper: bool = True,
        exchange: str = 'bybit',
    ) -> OptionTrade:
        """
        Open a new option trade.
        
        Args:
            signal_type: Signal that triggered trade (BULL_PROBE, PRIMARY_SHORT, etc.)
            direction: LONG or SHORT
            symbol: Option symbol
            qty: Position size
            entry_price: Option price at entry
            entry_spot: Underlying price at entry
            entry_iv: IV at entry (optional, will try to fetch from snapshot)
            notional: USD value at entry (computed if not provided)
            entry_timestamp: Explicit entry time (for historical backfill). Uses now if None.
            is_paper: Paper trade flag
            exchange: Exchange name
        
        Returns:
            Created OptionTrade instance
        """
        timestamp = entry_timestamp or datetime.now(timezone.utc)
        trade_id = f"{signal_type}_{timestamp.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        
        # Try to find matching snapshot for entry
        entry_snapshot = OptionSnapshot.objects.filter(
            symbol=symbol,
            exchange=exchange,
            timestamp__lte=timestamp,
        ).order_by('-timestamp').first()
        
        # Get IV from snapshot if not provided
        if entry_iv is None and entry_snapshot:
            entry_iv = entry_snapshot.iv
        
        # Compute notional if not provided
        if notional is None:
            notional = entry_price * qty
        
        trade = OptionTrade.objects.create(
            trade_id=trade_id,
            signal_type=signal_type,
            direction=direction,
            entry_timestamp=timestamp,
            entry_snapshot=entry_snapshot,
            entry_price=entry_price,
            entry_spot=entry_spot,
            entry_iv=entry_iv,
            symbol=symbol,
            qty=qty,
            notional=notional,
            is_paper=is_paper,
            exchange=exchange,
        )
        
        return trade
    
    def close_trade(
        self,
        trade_id: str,
        exit_price: Decimal,
        exit_spot: Decimal,
        exit_reason: str,
        exit_iv: Optional[Decimal] = None,
        exit_timestamp: Optional[datetime] = None,
    ) -> OptionTrade:
        """
        Close an existing trade and compute PnL attribution.
        
        Args:
            trade_id: Trade identifier
            exit_price: Option price at exit
            exit_spot: Underlying price at exit
            exit_reason: Reason for exit (tp, stop, time_stop, manual)
            exit_iv: IV at exit (optional, will try to fetch from snapshot)
            exit_timestamp: Explicit exit time (for historical backfill). Uses now if None.
        
        Returns:
            Updated OptionTrade instance
        """
        trade = OptionTrade.objects.get(trade_id=trade_id)
        
        if trade.exit_timestamp is not None:
            raise ValueError(f"Trade {trade_id} already closed")
        
        timestamp = exit_timestamp or datetime.now(timezone.utc)
        
        # Validate exit is after entry
        if timestamp <= trade.entry_timestamp:
            raise ValueError(f"Exit timestamp must be after entry timestamp")
        
        # Try to find matching snapshot for exit
        exit_snapshot = OptionSnapshot.objects.filter(
            symbol=trade.symbol,
            exchange=trade.exchange,
            timestamp__lte=timestamp,
        ).order_by('-timestamp').first()
        
        # Get IV from snapshot if not provided
        if exit_iv is None and exit_snapshot:
            exit_iv = exit_snapshot.iv
        
        # Compute realized PnL
        if trade.direction == 'LONG':
            realized_pnl = (exit_price - trade.entry_price) * trade.qty
        else:
            realized_pnl = (trade.entry_price - exit_price) * trade.qty
        
        # Compute PnL percentage
        pnl_pct = float(realized_pnl / trade.notional) if trade.notional else None
        
        # Compute path statistics (MAE/MFE)
        mae, mfe = self._compute_excursions(trade, timestamp)
        
        # Compute attribution
        iv_change = None
        if exit_iv is not None and trade.entry_iv is not None:
            iv_change = float(exit_iv - trade.entry_iv)
        
        spot_change_pct = None
        if trade.entry_spot and trade.entry_spot > 0:
            spot_change_pct = float((exit_spot - trade.entry_spot) / trade.entry_spot)
        
        # Update trade
        trade.exit_timestamp = timestamp
        trade.exit_snapshot = exit_snapshot
        trade.exit_price = exit_price
        trade.exit_spot = exit_spot
        trade.exit_iv = exit_iv
        trade.exit_reason = exit_reason
        trade.realized_pnl = realized_pnl
        trade.pnl_pct = pnl_pct
        trade.max_adverse_excursion = mae
        trade.max_favorable_excursion = mfe
        trade.iv_change = iv_change
        trade.spot_change_pct = spot_change_pct
        trade.save()
        
        return trade
    
    def _compute_excursions(
        self,
        trade: OptionTrade,
        exit_time: datetime,
    ) -> tuple[Optional[float], Optional[float]]:
        """
        Compute MAE and MFE from snapshots during hold period.
        
        MAE = max adverse excursion (worst unrealized loss %)
        MFE = max favorable excursion (best unrealized gain %)
        """
        # Get all snapshots during hold period
        snapshots = OptionSnapshot.objects.filter(
            symbol=trade.symbol,
            exchange=trade.exchange,
            timestamp__gt=trade.entry_timestamp,
            timestamp__lte=exit_time,
        ).order_by('timestamp')
        
        if not snapshots.exists():
            return None, None
        
        entry_price = float(trade.entry_price)
        is_long = trade.direction == 'LONG'
        
        max_favorable = 0.0
        max_adverse = 0.0
        
        for snap in snapshots:
            # Use mid price if available, else mark
            price = snap.mid_price or snap.mark_price
            if price is None:
                continue
            
            price = float(price)
            
            if is_long:
                pnl_pct = (price - entry_price) / entry_price
            else:
                pnl_pct = (entry_price - price) / entry_price
            
            if pnl_pct > 0:
                max_favorable = max(max_favorable, pnl_pct)
            else:
                max_adverse = max(max_adverse, abs(pnl_pct))
        
        return max_adverse if max_adverse > 0 else None, max_favorable if max_favorable > 0 else None
    
    def update_excursions(self, trade: OptionTrade) -> OptionTrade:
        """
        Update MAE/MFE for an open trade based on current snapshots.
        Call periodically to track path during hold.
        """
        if trade.exit_timestamp is not None:
            return trade  # Already closed
        
        now = datetime.now(timezone.utc)
        mae, mfe = self._compute_excursions(trade, now)
        
        # Only update if we found worse/better values
        if mae is not None:
            if trade.max_adverse_excursion is None or mae > trade.max_adverse_excursion:
                trade.max_adverse_excursion = mae
        
        if mfe is not None:
            if trade.max_favorable_excursion is None or mfe > trade.max_favorable_excursion:
                trade.max_favorable_excursion = mfe
        
        trade.save()
        return trade
    
    def get_open_trades(self) -> list[OptionTrade]:
        """Get all open trades."""
        return list(OptionTrade.objects.filter(exit_timestamp__isnull=True))
    
    def get_trade_summary(self, trade: OptionTrade) -> dict:
        """Get summary dict for a trade."""
        return {
            'trade_id': trade.trade_id,
            'signal_type': trade.signal_type,
            'direction': trade.direction,
            'symbol': trade.symbol,
            'entry_timestamp': trade.entry_timestamp,
            'entry_price': float(trade.entry_price),
            'entry_spot': float(trade.entry_spot),
            'entry_iv': float(trade.entry_iv) if trade.entry_iv else None,
            'exit_timestamp': trade.exit_timestamp,
            'exit_price': float(trade.exit_price) if trade.exit_price else None,
            'exit_spot': float(trade.exit_spot) if trade.exit_spot else None,
            'exit_iv': float(trade.exit_iv) if trade.exit_iv else None,
            'exit_reason': trade.exit_reason,
            'realized_pnl': float(trade.realized_pnl) if trade.realized_pnl else None,
            'pnl_pct': trade.pnl_pct,
            'mae': trade.max_adverse_excursion,
            'mfe': trade.max_favorable_excursion,
            'iv_change': trade.iv_change,
            'spot_change_pct': trade.spot_change_pct,
            'is_open': trade.exit_timestamp is None,
        }
