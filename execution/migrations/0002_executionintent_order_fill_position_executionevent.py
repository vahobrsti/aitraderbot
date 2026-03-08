from django.db import migrations, models
import django.db.models.deletion
import django.utils.timezone
import uuid


class Migration(migrations.Migration):

    dependencies = [
        ('signals', '0001_initial'),
        ('execution', '0001_initial'),
    ]

    operations = [
        migrations.CreateModel(
            name='ExecutionIntent',
            fields=[
                ('id', models.UUIDField(default=uuid.uuid4, editable=False, primary_key=True, serialize=False)),
                ('signal_date', models.DateField(db_index=True)),
                ('direction', models.CharField(choices=[('long', 'Long'), ('short', 'Short')], max_length=10)),
                ('instrument_type', models.CharField(default='option', help_text='option, perpetual, future', max_length=20)),
                ('target_symbol', models.CharField(blank=True, max_length=50)),
                ('target_qty', models.DecimalField(blank=True, decimal_places=8, max_digits=18, null=True)),
                ('target_notional_usd', models.DecimalField(blank=True, decimal_places=2, max_digits=12, null=True)),
                ('option_type', models.CharField(blank=True, help_text='call or put', max_length=10)),
                ('strike_price', models.DecimalField(blank=True, decimal_places=2, max_digits=12, null=True)),
                ('expiry_date', models.DateField(blank=True, null=True)),
                ('stop_loss_pct', models.DecimalField(blank=True, decimal_places=4, max_digits=5, null=True)),
                ('take_profit_pct', models.DecimalField(blank=True, decimal_places=4, max_digits=5, null=True)),
                ('max_hold_days', models.IntegerField(blank=True, null=True)),
                ('status', models.CharField(choices=[('pending', 'Pending'), ('risk_check', 'Risk Check'), ('approved', 'Approved'), ('rejected', 'Rejected'), ('executing', 'Executing'), ('partial', 'Partially Filled'), ('filled', 'Filled'), ('cancelled', 'Cancelled'), ('failed', 'Failed')], default='pending', max_length=20)),
                ('status_reason', models.TextField(blank=True)),
                ('idempotency_key', models.CharField(help_text='Unique key to prevent duplicate executions', max_length=100, unique=True)),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('updated_at', models.DateTimeField(auto_now=True)),
                ('approved_at', models.DateTimeField(blank=True, null=True)),
                ('completed_at', models.DateTimeField(blank=True, null=True)),
                ('account', models.ForeignKey(on_delete=django.db.models.deletion.PROTECT, related_name='intents', to='execution.exchangeaccount')),
                ('signal', models.ForeignKey(on_delete=django.db.models.deletion.PROTECT, related_name='execution_intents', to='signals.dailysignal')),
            ],
            options={
                'db_table': 'execution_intent',
                'ordering': ['-created_at'],
            },
        ),
        migrations.AddIndex(
            model_name='executionintent',
            index=models.Index(fields=['signal_date', 'status'], name='execution_i_signal__abc123_idx'),
        ),
        migrations.AddIndex(
            model_name='executionintent',
            index=models.Index(fields=['account', 'status'], name='execution_i_account_def456_idx'),
        ),
    ]
