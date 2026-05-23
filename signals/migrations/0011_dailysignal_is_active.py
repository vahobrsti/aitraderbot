# signals/migrations/0011_dailysignal_is_active.py
"""
Add is_active field to DailySignal for manual operator deactivation.

When an operator deactivates a signal, it becomes invisible to tradeable()
and active() queries, but the row is preserved. If the same trade type
re-qualifies on a later hourly run, it can be reactivated.
"""

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('signals', '0010_unique_date_trade_decision'),
    ]

    operations = [
        migrations.AddField(
            model_name='dailysignal',
            name='is_active',
            field=models.BooleanField(
                default=True,
                db_index=True,
                help_text='False = manually deactivated by operator. Currently always True (signals are final once fired).',
            ),
        ),
    ]
