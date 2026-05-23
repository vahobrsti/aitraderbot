# signals/migrations/0010_unique_date_trade_decision.py
"""
Allow multiple signals per day by changing the unique constraint from
`date` alone to `(date, trade_decision)`.

This enables independent trade types (e.g., MVRV_SHORT + IRON_CONDOR)
to coexist on the same date, matching backtest behavior.
"""

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('signals', '0009_add_condor_strike_fields'),
    ]

    operations = [
        # 1. Remove unique=True from date field (keep index)
        migrations.AlterField(
            model_name='dailysignal',
            name='date',
            field=models.DateField(db_index=True),
        ),
        # 2. Add is_active field (default True for existing rows)
        migrations.AddField(
            model_name='dailysignal',
            name='is_active',
            field=models.BooleanField(
                default=True,
                db_index=True,
                help_text='False = signal no longer qualifies but couldn\'t be deleted (protected by execution intent)',
            ),
        ),
        # 3. Add composite unique constraint (date, trade_decision)
        migrations.AddConstraint(
            model_name='dailysignal',
            constraint=models.UniqueConstraint(
                fields=['date', 'trade_decision'],
                name='unique_date_trade_decision',
            ),
        ),
        # 4. Update ordering
        migrations.AlterModelOptions(
            name='dailysignal',
            options={
                'ordering': ['-date', '-updated_at'],
                'verbose_name': 'Daily Signal',
                'verbose_name_plural': 'Daily Signals',
            },
        ),
    ]
